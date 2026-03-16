<div align="center">
<!-- <h1> LBGD </h1> -->
<h2> <a href="">Localized Background-aware Generative Distillation for Enhanced Remote Sensing Object Detection</h2>
<h3> Chao Wang, Yanguang Sun, Jian Yang, Lei Luo*</h3>
<h4> Accepted to IEEE TCSVT 2026</h4>
</div>


## ✨ Overview
<p align="center"> <img src="https://github.com/wchao0601/LBGD/blob/master1/LBGD-Network.png" width="99.5%"> </p>



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
@article{wang2026localized,
  title={Localized Background-aware Generative Distillation for Enhanced Remote Sensing Object Detection},
  author={Wang, Chao and Sun, Yanguang and Yang, Jian and Luo, Lei},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2026},
  publisher={IEEE}
}

@article{wang2025msod,
  title={MSOD: A Large-Scale Multi-Scene Dataset and a Novel Diagonal-Geometry Loss for SAR Object Detection},
  author={Wang, Chao and Fang, Wenxuan and Li, Xiang and Yang, Jian and Luo, Lei},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025},
  publisher={IEEE}
}

@article{wang2025m4,
  title={M4-SAR: A Multi-Resolution, Multi-Polarization, Multi-Scene, Multi-Source Dataset and Benchmark for Optical-SAR Fusion Object Detection},
  author={Wang, Chao and Lu, Wei and Li, Xiang and Yang, Jian and Luo, Lei},
  journal={arXiv preprint arXiv:2505.10931},
  year={2025}
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

