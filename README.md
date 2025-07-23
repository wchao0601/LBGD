<div align="center">
<!-- <h1> LBGD </h1> -->
<h2> <a href="">Localized Background-aware Generative Distillation for Enhanced Remote Sensing Object Detection</h2>
<h4> 2025</h4>
</div>


## ✨ Overview
Feature-based knowledge distillation has attracted significant attention in remote sensing object detection. The main challenge in this method is that feature distillation may misguide the detection of tiny remote-sensing objects due to the lack of local background priors. To address this issue, this paper proposes the Localized Background-aware Generative Distillation (LBGD) method, which incorporates two key components: the lightweight diffusion reconstructor (LDR) and the patch-wise channel distillation  (PCD) loss. LDR dynamically adjusts the receptive field to effectively capture the local background information surrounding the target. Meanwhile, PCD emphasizes the most salient patch regions in each channel, reducing the impact of global background information. To the best of our knowledge, localized background-aware generative distillation mechanisms have not been previously explored in remote sensing object detection. Numerous experimental results demonstrate that LBGD brings significant performance improvements, for example, SODA-A (+1.9\% $mAP$), and DIOR (+2.8\% $mAP$).
<p align="center"> <img src="https://github.com/wchao0601/LBGD/blob/master1/LBGD-Network.png" width="99.5%"> </p>
<div align="center"; style="text-align: center; display: flex; justify-content: space-between;">
    <img src="https://github.com/wchao0601/LBGD/blob/master1/DRM.png" alt="Image 1" style="width: 49%;" />
    <img src="https://github.com/wchao0601/LBGD/blob/master1/PCD.png" alt="Image 2" style="width: 49%;" />
</div>



## 📄 Documentation
### Installation
Create and activate a conda environment:
```
conda create -n lbgd python=3.11
conda activate lbgd
```
Install the required packages:
```
git clone https://github.com/wchao0601/LBGD.git
cd LBGD/
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install seaborn thop timm einops
pip install -r requirements.txt
```


### Data Preparation

| Dataset | Down-Link | Image Size |
| :---: | :---: | :---: |
| DIOR-R | [Baidu](https://pan.baidu.com/s/1uLwBVIG2NPUSrAxXbVUt3g?pwd=0601)|800 x 800|


### Train
```python
python train.py
```

### Test
```python
python test.py
```

### Predict
```python
python predict.py
```

## 📈 Results
<p align="center"> <img src="https://github.com/wchao0601/LBGD/blob/master1/results.png" width="99.5%"> </p>
<p align="center"> <img src="https://github.com/wchao0601/LBGD/blob/master1/detect-vis1.png" width="99.5%"> </p>

|  Model     | YOLOv8S | YOLOv8N | Mimic | CWD | MGD | PKD | CrossKD | LSKD | LBGD (Ours)|
| :---:      | :---: | :---:| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  Weights   |[Download](https://pan.baidu.com/s/1wLSLH8xxIRUmrbhCGv59bQ?pwd=0601)|[Download](https://pan.baidu.com/s/1yHA6ER-iT2gV-KMYh9XKjQ?pwd=0601)|[Download](https://pan.baidu.com/s/1s10Dgkc4AHA0ERQI0MynNg?pwd=0601)|[Download](https://pan.baidu.com/s/1gl8I_uJENIfLNPCIBMWmIg?pwd=0601)|[Download](https://pan.baidu.com/s/1Iz16XkU_JB8PBKTMfxWnSA?pwd=0601)|[Download](https://pan.baidu.com/s/1B4gMxwclgogyCosHkYla3A?pwd=0601)|[Download](https://pan.baidu.com/s/1_UvWYoRsh15UYMBkWJYeAg?pwd=0601)|[Download](https://pan.baidu.com/s/18w-6Z23H40oRO_uCRgIXhQ?pwd=0601)|[Download](https://pan.baidu.com/s/16aKLoscwT10lvOEXE0Pvuw?pwd=0601)|

## 🌐 Contact
If you have any questions, please feel free to contact me via email at wchao0601@163.com

## 📚 Citation
If our work is helpful, you can cite our paper:
```
@article{wang2025m4,
  title={M4-SAR: A Multi-Resolution, Multi-Polarization, Multi-Scene, Multi-Source Dataset and Benchmark for Optical-SAR Fusion Object Detection},
  author={Wang, Chao and Lu, Wei and Li, Xiang and Yang, Jian and Luo, Lei},
  journal={arXiv preprint arXiv:2505.10931},
  year={2025}
}

@inproceedings{wang2025cross,
  title={Cross-modal Gaussian Localization Distillation for Optical Information guided SAR Object Detection},
  author={Wang, Chao and Luo, Lei and Fang, Wenxuan and Yang, Jian},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}

@inproceedings{wang2024psekd,
  title={Psekd: Phase-shift encoded knowledge distillation for oriented object detection in remote sensing images},
  author={Wang, Chao and Yue, Yubiao and Luo, Bingchun and Chen, Yujie and Xue, Jun},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={2680--2684},
  year={2024},
  organization={IEEE}
}

@article{wang2023category,
  title={Category-oriented localization distillation for sar object detection and a unified benchmark},
  author={Wang, Chao and Ruan, Rui and Zhao, Zhicheng and Li, Chenglong and Tang, Jin},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={61},
  pages={1--14},
  year={2023},
  publisher={IEEE}
}

```
## 🙏 Acknowledgment
- This repo is based on [Ultralytics](https://github.com/ultralytics/ultralytics) and [MGD](https://github.com/yzd-v/MGD) which are excellent works.
- We thank the [YOLOv12](https://github.com/sunsmarterjie/yolov12) libraries, which help us to implement our ideas quickly.

