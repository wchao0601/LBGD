<div align="center">
<!-- <h1> LBGD </h1> -->
<h2> <a href="">Localized Background-aware Generative Distillation for Enhanced Remote Sensing Object Detection</h2>
<h4> 2025</h4>
</div>


## ✨ Overview
Feature-based knowledge distillation has attracted significant attention in remote sensing object detection. The main challenge in this method is that feature distillation may misguide the detection of tiny remote-sensing objects due to the lack of local background priors. To address this issue, this paper proposes the Localized Background-aware Generative Distillation (LBGD) method, which incorporates two key components: the lightweight diffusion reconstructor (LDR) and the patch-wise channel distillation loss (PCD). LDR dynamically adjusts the receptive field to effectively capture the local background information surrounding the target. Meanwhile, PCD emphasizes the most salient patch regions in each channel, reducing the impact of global background information. To the best of our knowledge, localized background-aware generative distillation mechanisms have not been previously explored in remote sensing object detection. Numerous experimental results demonstrate that LBGD brings significant performance improvements, for example, SODA-A (+1.9\% $mAP$), and DIOR (+2.8\% $mAP$).



## 📄 Documentation
### Installation
Create and activate a conda environment:
```
conda create -n lbgd python=3.11
conda activate lbgd
```
Install the required packages:
```
git clone .../LBGD.git
cd LBGD/
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1
pip install seaborn thop timm einops
pip install -r requirements.txt
```


### Data Preparation

| Dataset | Image Size |
| :---: | :---: |
| DIOR-R |800 x 800|


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

