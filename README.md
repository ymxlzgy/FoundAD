
# FoundAD
---

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Dataset Preparation](#dataset-preparation)
3. [Quick Start](#quick-start)
   - [Few-Shot Sampling](#few-shot-sampling)
   - [Model Training](#model-training)
   - [Anomaly Detection / Inference](#anomaly-detection--inference)
   
---

## Environment Setup

All Python dependencies are listed in **`requirements.txt`**. We recommend **Python ≥ 3.10**.

```bash
conda create -n foundad python=3.10
conda activate foundad
pip install -r requirements.txt
pip install -e .
```


## Dataset Preparation

FoundAD supports the following anomaly-detection benchmarks:

| Dataset | Preferred download |
|---------|--------------------|
| **MVTec AD** | Official site: <https://www.mvtec.com/company/research/datasets/mvtec-ad> |
| **VisA** | We use the structured dataset of RealNet: <https://github.com/cnulab/RealNet> |

---

## Quick Start

### Few-Shot Sampling

Create a **few-shot** subset with `sample.py`:

```bash
python foundad/src/sample.py source=/media/ymxlzgy/Data21/xinyan/visa target=/media/ymxlzgy/Data21/xinyan/visa_tmp num_samples=2
```
`source` is the dataset folder. `target` is the folder of few-shot samples.

### Model Training

Train:

```bash
python foundad/main.py mode=train data.batch_size=8 data.dataset=mvtec data.data_name=mvtec_1shot data.data_path=/media/ymxlzgy/Data21/xinyan app=train_dinov2 diy_name=dbug
```
`data.dataset` is "mvtec" or "visa". `data.data_name` is the folder name of few-shot samples. `data.data_path` is the path where the few-shot folder is at. `app` is train_dinov2 or train_dinov3 under configs/app/. `diy_name` (optionally) is the post-fix name of the model saving directory.

### Anomaly Detection / Inference

After training, run inference:

```bash
python foundad/main.py mode=AD data.dataset=mvtec data.data_name=mvtec_1shot data.test_root=/media/ymxlzgy/Data21/xinyan/mvtec diy_name=dbug app=test_dinov3 app.ckpt_step=1950
```
 `data.test_root` is the dataset folder. `app` is test_dinov2 or test_dinov3 under configs/app/.


---

