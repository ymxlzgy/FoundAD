
# FoundAD <img src="./assets/icon.png" alt="Logo" width="30" height="30">

This repo contains implementation of the paper **Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors**, [arXiv]().

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Quick Start](#quick-start)
3. [Training and Inference](#train-infer)
   - [Dataset Preparation](#dataset-preparation)
   - [Few-Shot Sampling](#few-shot-sampling)
   - [Model Training](#model-training)
   - [Anomaly Detection / Inference](#anomaly-detection--inference)
4. [Acknowledgement](#acknowledgement)
   

## Environment Setup

All Python dependencies are listed in `requirements.txt`. We recommend Python ≥ 3.10.

```bash
conda create -n foundad python=3.10
conda activate foundad
git clone git@github.com:ymxlzgy/FoundAD.git
cd FoundAD
pip install -r requirements.txt
pip install -e .
```


## Quick Start
Before we start, please make sure you have the rights to use [DINOv3](https://github.com/facebookresearch/dinov3). Download our trained weights based on DINOv3 ViT-B/16, and put them to `./logs/`. 
||1-shot|2-shot|4-shot|8-shot|
|---------|:---------:|:---------:|:---------:|:---------:|
|**MVTec AD**|[⬇️ <u>link</u>](https://www.campar.in.tum.de/public_datasets/2025_foundad/mvtec_1shot.zip)|[⬇️ <u>link</u>](https://www.campar.in.tum.de/public_datasets/2025_foundad/mvtec_2shot.zip)|[⬇️ <u>link</u>](https://www.campar.in.tum.de/public_datasets/2025_foundad/mvtec_4shot.zip)|[⬇️ <u>link</u>](https://www.campar.in.tum.de/public_datasets/2025_foundad/mvtec_8shot.zip)|
|**VisA**  |[⬇️ <u>link</u>](https://www.campar.in.tum.de/public_datasets/2025_foundad/visa_1shot.zip)|[⬇️ <u>link</u>](https://www.campar.in.tum.de/public_datasets/2025_foundad/visa_2shot.zip)|[⬇️ <u>link</u>](https://www.campar.in.tum.de/public_datasets/2025_foundad/visa_4shot.zip)|[⬇️ <u>link</u>](https://www.campar.in.tum.de/public_datasets/2025_foundad/visa_8shot.zip)|


Run
```bash
python foundad/main.py mode=AD testing.segmentation_vis=True data.dataset=mvtec data.data_name=mvtec_1shot data.test_root=assets/mvtec
```

Or
```bash
python foundad/main.py mode=AD testing.segmentation_vis=True data.dataset=visa data.data_name=visa_1shot data.test_root=assets/visa
```


## Training and Inference

### Dataset Preparation

| Dataset | Preferred download |
|---------|--------------------|
| **MVTec AD** | Official site: [<u>Here</u>](https://www.mvtec.com/company/research/datasets/mvtec-ad) |
| **VisA** | We use the structured dataset of [<u>RealNet</u>](https://github.com/cnulab/RealNet). |

### Few-Shot Sampling

Create a **few-shot** subset with `sample.py`:

```bash
python foundad/src/sample.py source=/media/ymxlzgy/Data21/xinyan/visa target=/media/ymxlzgy/Data21/xinyan/visa_tmp num_samples=2
```
where `source` is the dataset folder, and `target` is the folder of few-shot samples. We also provide the data we sampled and trained the models on: [<u>⬇️ mvtec-few-shot</u>](https://www.campar.in.tum.de/public_datasets/2025_foundad/few-shot-mvtec.zip) and [<u>⬇️ visa-few-shot</u>](https://www.campar.in.tum.de/public_datasets/2025_foundad/few-shot-visa.zip).

### Model Training

```bash
python foundad/main.py mode=train data.batch_size=8 data.dataset=mvtec data.data_name=mvtec_1shot data.data_path=/media/ymxlzgy/Data21/xinyan app=train_dinov2 diy_name=dbug
```
where `data.dataset` is "mvtec" or "visa", `data.data_name` is the folder name of few-shot samples, `data.data_path` is the path where the few-shot folder is at, `app` is "train_dinov2" or "train_dinov3" under `configs/app/`, and `diy_name` (optionally) is the post-fix name of the model saving directory. To adjust the layer, please specify `app.meta.n_layer`.

### Anomaly Detection / Inference

After training, run inference:

```bash
python foundad/main.py mode=AD data.dataset=mvtec data.data_name=mvtec_1shot diy_name=dbug data.test_root=/media/ymxlzgy/Data21/xinyan/mvtec app=test app.ckpt_step=1950
```
where `data.test_root` is the dataset folder, and `app` is test_dinov2 or test_dinov3 under `configs/app/`. To adjust sample number K, please specify `testing.K_top_mvtec` and `testing.K_top_visa`.

## Acknowledgement
This repo utilizes [DINOv3](https://github.com/facebookresearch/dinov3), [DINOv2](https://github.com/facebookresearch/dinov2), [DINO](https://github.com/facebookresearch/dino), [SigLIP](https://github.com/google-research/big_vision), [CLIP](https://github.com/openai/CLIP) and [WideResNet](https://pytorch.org/hub/pytorch_vision_wide_resnet/). We also thank [I-JEPA](https://github.com/facebookresearch/ijepa) for the inspiration.