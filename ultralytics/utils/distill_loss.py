
import torch
import torch.nn as nn 
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from .tal import bbox2dist
from .loss import v8DetectionLoss, BboxLoss, RotatedBboxLoss
from torch.cuda.amp import autocast
import torchvision.utils as vutils
import logging
logger = logging.getLogger(__name__)

USE_FLASH_ATTN = False
try:
    import torch
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
        from flash_attn.flash_attn_interface import flash_attn_func
        USE_FLASH_ATTN = True
    else:
        from torch.nn.functional import scaled_dot_product_attention as sdpa
        logger.warning("FlashAttention is not available on this device. Using scaled_dot_product_attention instead.")
except Exception:
    from torch.nn.functional import scaled_dot_product_attention as sdpa
    logger.warning("FlashAttention is not available on this device. Using scaled_dot_product_attention instead.")
    
class MimicLoss(nn.Module):
    def __init__(self, channels_s, channels_t):
        super(MimicLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mse = nn.MSELoss()

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            losses.append(self.mse(s, t))
        loss = sum(losses)
        return loss

class CWDLoss(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.
    <https://arxiv.org/abs/2011.13256>`_.
    """

    def __init__(self, channels_s, channels_t,tau=1.0):
        super(CWDLoss, self).__init__()
        self.tau = tau

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):

            assert s.shape == t.shape
            
            N, C, H, W = s.shape
            
            # normalize in channel diemension
            softmax_pred_T = F.softmax(t.view(-1, W * H) / self.tau, dim=1)  # [N*C, H*W]
            
            logsoftmax = torch.nn.LogSoftmax(dim=1)
            cost = torch.sum(
                softmax_pred_T * logsoftmax(t.view(-1, W * H) / self.tau) - 
                softmax_pred_T * logsoftmax(s.view(-1, W * H) / self.tau)) * (self.tau ** 2)

            losses.append(cost / (C * N))
        loss = sum(losses)

        return loss
    
class MGDLoss(nn.Module):
    def __init__(self, channels_s, channels_t, alpha_mgd=0.00002, lambda_mgd=0.65):
        super(MGDLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd

        self.generation = [
            nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, kernel_size=3, padding=1)).to(device) for channel in channels_t
        ]

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            losses.append(self.get_dis_loss(s, t, idx) * self.alpha_mgd)
        loss = sum(losses)
        return loss

    def get_dis_loss(self, preds_S, preds_T, idx):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N, 1, H, W)).to(device)
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation[idx](masked_fea)

        dis_loss = loss_mse(new_fea, preds_T) / N

        return dis_loss

def norm(feat: torch.Tensor) -> torch.Tensor:
    """Normalize the feature maps to have zero mean and unit variances.

    Args:
        feat (torch.Tensor): The original feature map with shape
            (N, C, H, W).
    """
    assert len(feat.shape) == 4
    N, C, H, W = feat.shape
    feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
    mean = feat.mean(dim=-1, keepdim=True)
    std = feat.std(dim=-1, keepdim=True)
    feat = (feat - mean) / (std + 1e-6)
    return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)


class PKDLoss(nn.Module):
    def __init__(self, channels_s, channels_t):
        super(PKDLoss, self).__init__()

    def forward(self, y_s, y_t):
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape
            s = norm(s)
            t = norm(t)
            losses.append(F.mse_loss(s, t, reduction='none').mean() / 2)
        loss = sum(losses)

        return loss

class LightDiffusionReconstructor(nn.Module):
    def __init__(self, channels, num_steps=3, hidden_dim=32):
        """
        Lightweight Diffusion Reconstructor (LDR)
        Parameters:
            channels: Number of input and output channels (same)
            num_steps: Number of diffusion steps (default: 3 steps)
            hidden_dim: Hidden layer dimension (default: 32)
        """
        super().__init__()
        self.num_steps = num_steps
        self.channels = channels
        
        # Shared Base Module
        self.base_block = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 1),  # lightweight channel adjustment
            nn.GroupNorm(4, hidden_dim),
            nn.SiLU()
        )
        
        # Multi-scale Receptive-field Expansion Module (MREM)
        self.context_block = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, dilation=1),
            nn.GroupNorm(4, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2),  # expand the receptive field
            SpatialAttention(hidden_dim)
        )
        
        # Add Self-Attention after Noise
        self.self_attn = A_Attn(hidden_dim, area=8)

        # Diffusion-based Reconstruction Module (DRM)
        self.diffusion_step = nn.Sequential(
            DepthwiseSeparableConv(hidden_dim*2, hidden_dim),
            nn.GroupNorm(4, hidden_dim),
            nn.SiLU(),
            DepthwiseSeparableConv(hidden_dim, hidden_dim),
            nn.GroupNorm(4, hidden_dim)
        )
        
        # Noise Parameters
        self.noise_scheduler = nn.Parameter(torch.linspace(0.1, 0.01, num_steps))
        
        # Final Output
        self.final_proj = nn.Conv2d(hidden_dim, channels, 1)
    
    def forward(self, feat_a):
        B, C, H, W = feat_a.shape
        
        # Initial Feature Extraction
        x = self.base_block(feat_a)
        context = self.context_block(x)

        # Diffusion Reconstruction Process
        for step in range(self.num_steps):

            # Noise Injection
            noise_level = self.noise_scheduler[step]
            noise = torch.randn_like(x) * noise_level
            x_noisy = x + noise
            x_noisy = self.self_attn(x_noisy)

            # Concatenate Contextual Information
            x_input = torch.cat([x_noisy, context], dim=1)
            residual = self.diffusion_step(x_input)
            
            # Feature Correction
            x = x + residual  # Residual Update
            
            # No noise is added in the final step
            if step == self.num_steps - 2:
                x = x.detach()  # Cut off the noise gradient in the final step
        
        # Final Output
        return self.final_proj(x) + feat_a 

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class A_Attn(nn.Module):
    """
    Area-attention module with the requirement of flash attention.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1.

    Methods:
        forward: Performs a forward process of input tensor and outputs a tensor after the execution of the area attention mechanism.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import AAttn
        >>> model = AAttn(dim=64, num_heads=2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    
    Notes: 
        recommend that dim//num_heads be a multiple of 32 or 64.

    """

    def __init__(self, dim, num_heads=1, area=4):
        """Initializes the area-attention module, a simple yet efficient attention module for YOLO."""
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.qkv1 = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)


    def forward(self, x):
        """Processes the input tensor 'x' through the area-attention"""
        B1, C1, H1, W1 = x.shape
        if H1 % self.area != 0:
            pad1 = self.area - (H1 % self.area)
            x = F.pad(x, pad=(0, pad1, 0, pad1))
            B, C, H, W = x.shape
            N = H * W
        else:
            B, C, H, W = x.shape
            N = H * W
        qkv = self.qkv(x).flatten(2).transpose(1, 2)



        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = qkv.view(B, N, self.num_heads, self.head_dim * 3).split(
            [self.head_dim, self.head_dim, self.head_dim], dim=3
        )

        if x.is_cuda and USE_FLASH_ATTN:
            x = flash_attn_func(
                q.contiguous().half(),
                k.contiguous().half(),
                v.contiguous().half()
            ).to(q.dtype)
        elif x.is_cuda and not USE_FLASH_ATTN:
            x = sdpa(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), attn_mask=None, dropout_p=0.0, is_causal=False)
            x = x.permute(0, 2, 1, 3)
        else:
            q = q.permute(0, 2, 3, 1)
            k = k.permute(0, 2, 3, 1)
            v = v.permute(0, 2, 3, 1)
            attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
            max_attn = attn.max(dim=-1, keepdim=True).values 
            exp_attn = torch.exp(attn - max_attn)
            attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
            x = (v @ attn.transpose(-2, -1))
            x = x.permute(0, 3, 1, 2)
            v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        x = x + self.pe(v)
        x = self.proj(x)

        x = x[:, :, :H1, :W1]
        return x


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution to Reduce Parameters
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)
    
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class SpatialAttention(nn.Module):
    """
    Lightweight Spatial Attention to Expand Receptive Field
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        attn = self.sigmoid(self.conv(x))
        return x * attn + x 

    
class LBGDLoss(nn.Module):

    """
    Localized Background-aware Generative Distillation for Enhanced Remote Sensing Object Detection
    """

    def __init__(self, channels_s, channels_t, tau=1.5, path_size=2):
        super(LBGDLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tau = tau
        self.generation = [LightDiffusionReconstructor(channels=channel).to(device) for channel in channels_t]
        self.path_size = path_size
        self.path_area = path_size * path_size

    def forward(self, y_s, y_t):
        """Forward computation.
        Args:
            y_s (list): The student model prediction with
                shape (N, C, H, W) in list.
            y_t (list): The teacher model prediction with
                shape (N, C, H, W) in list.
        Return:
            torch.Tensor: The calculated loss value of all stages.
        """
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            assert s.shape == t.shape

            N, C, H, W = s.shape
            pad = 0

            if H % self.path_size != 0:
                while (H + pad) % self.path_size != 0:
                    pad = pad + 1
                padding = (0, pad, 0, pad)
                s = F.pad(s, padding, mode='constant', value=0)
                t = F.pad(t, padding, mode='constant', value=0)

            s = self.generation[idx](s) # LDR

            softmax_pred_T = F.softmax(t.view(N*C, (H+pad)*(W+pad)//self.path_area, self.path_size, self.path_size) / self.tau, dim=1)

            logsoftmax = torch.nn.LogSoftmax(dim=1)

            pcd = torch.sum( # PCD
                softmax_pred_T * logsoftmax(t.view(N*C, (H+pad)*(W+pad)//self.path_area, self.path_size, self.path_size) / self.tau) - 
                softmax_pred_T * logsoftmax(s.view(N*C, (H+pad)*(W+pad)//self.path_area, self.path_size, self.path_size) / self.tau)) * (self.tau ** 2)

            losses.append(pcd / (C * N))

        loss = sum(losses)

        return loss


class FeatureLoss(nn.Module):
    def __init__(self, channels_s, channels_t, kd_type='lbgd', loss_weight=1.0):
        super(FeatureLoss, self).__init__()
        self.loss_weight = loss_weight
      
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.align_module = nn.ModuleList([
            nn.Conv2d(channel, tea_channel, kernel_size=1, stride=1, padding=0).to(device)
            for channel, tea_channel in zip(channels_s, channels_t)
        ])
        self.norm = [
            nn.BatchNorm2d(tea_channel, affine=False).to(device)
            for tea_channel in channels_t
        ]
        
        if kd_type == 'mimic':
            self.feature_loss = MimicLoss(channels_s, channels_t)
        elif kd_type == 'cwd':
            self.feature_loss = CWDLoss(channels_s, channels_t)
        elif kd_type == 'mgd':
            self.feature_loss = MGDLoss(channels_s, channels_t)
        elif kd_type == 'pkd':
            self.feature_loss = PKDLoss(channels_s, channels_t) 
        elif kd_type == 'lbgd':
            self.feature_loss = LBGDLoss(channels_s, channels_t)
        else:
            raise NotImplementedError

    @autocast(True)
    def forward(self, y_s, y_t):
        assert len(y_s) == len(y_t)
        tea_feats = []
        stu_feats = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            s = self.align_module[idx](s)
            s = self.norm[idx](s)
            t = self.norm[idx](t)
            tea_feats.append(t)
            stu_feats.append(s)

        loss = self.feature_loss(stu_feats, tea_feats)
        return self.loss_weight * loss


class Distill_LogitLoss:
    def __init__(self,p, t_p, alpha =0.25):
        
        t_ft = torch.cuda.FloatTensor if t_p[0].is_cuda else torch.Tensor
        self.p =p
        self.t_p = t_p 
        self.logit_loss = t_ft([0])
        self.DLogitLoss = nn.MSELoss(reduction="none")
        self.bs = p[0].shape[0]
        self.alpha = alpha
    
    def __call__(self):
        # per output
        assert len(self.p) == len(self.t_p)
        for i, (pi,t_pi) in enumerate(zip(self.p,self.t_p)):  # layer index, layer predictions
            assert pi.shape == t_pi.shape
            self.logit_loss += torch.mean(self.DLogitLoss(pi, t_pi))
        return self.logit_loss[0]*self.alpha


def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)


def kd_quality_focal_loss(pred,
                          target,
                          weight=None,
                          beta=1,
                          reduction='mean',
                         ):
    num_classes = pred.size(1)
    if weight is not None:
        weight = weight[:, None].repeat(1, num_classes)

    target = target.detach().sigmoid()
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    focal_weight = torch.abs(pred.sigmoid() - target).pow(beta)
    loss = loss * focal_weight
    # loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

class v8OBBKDLoss(v8DetectionLoss):
    def __init__(self, model):
        """
        Initializes v8OBBLoss with model, assigner, and rotated bbox loss.

        Note model must be de-paralleled.
        """
        # super().__init__(model)
        device = next(model.parameters()).device 
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        m = model.model[-1]
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max - 1, use_dfl=self.use_dfl).to(self.device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds, preds_t, batch):
        """Calculate and return the loss for the YOLO model."""
        self.device = preds[1].device
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        loss_kd = torch.zeros(1, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        feats_t, pred_angle_t = preds_t if isinstance(preds_t[0], list) else preds_t[1] # teacher
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        # teacher
        pred_distri_t, pred_scores_t = torch.cat([xi.view(feats_t[0].shape[0], self.no, -1) for xi in feats_t], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()
        # teacher
        pred_scores_t = pred_scores_t.permute(0, 2, 1).contiguous()
        pred_distri_t = pred_distri_t.permute(0, 2, 1).contiguous()
        pred_angle_t = pred_angle_t.permute(0, 2, 1).contiguous()
        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        # teacher
        imgsz_t = torch.tensor(feats_t[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points_t, stride_tensor_t = make_anchors(feats_t, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        targets = targets.to(self.device)
        gt_labels = gt_labels.to(self.device)
        gt_bboxes = gt_bboxes.to(self.device)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)
        pred_bboxes_t = self.bbox_decode(anchor_points_t, pred_distri_t, pred_angle_t)  # xyxy, (b, h*w, 4)
        
        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        bboxes_for_assigner_t = pred_bboxes_t.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner_t[..., :4] *= stride_tensor_t
        _, target_bboxes_t, target_scores_t, fg_mask_t, _ = self.assigner(
            pred_scores_t.detach().sigmoid(),
            bboxes_for_assigner_t.type(gt_bboxes.dtype),
            anchor_points * stride_tensor_t,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        
        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= 7.5  # box gain
        loss[1] *= 0.5  # cls gain
        loss[2] *= 1.5  # dfl gain
        loss_kd += kd_quality_focal_loss(pred_scores.view(-1,pred_scores.shape[2]), pred_scores_t.view(-1,pred_scores.shape[2])).mean()# crosskd

        return (loss.sum() + loss_kd) * batch_size, loss.detach() 
    

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            device = pred_dist.device
            self.proj = self.proj.to(device)
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)

