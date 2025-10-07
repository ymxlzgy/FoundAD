
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
pip install -r requirements.txt
pip install -e .
```


## Dataset Preparation

ADJEPA supports the following anomaly-detection benchmarks:

| Dataset | Preferred download |
|---------|--------------------|
| **MVTec AD** | Official site: <https://www.mvtec.com/company/research/datasets/mvtec-ad> |
| **VISA** | : we recommend to use the structured dataset of RealNet: <https://github.com/cnulab/RealNet> |

---

## Quick Start

### Few-Shot Sampling

Create a **few-shot** subset with `sample.py`:

```bash
python ADJEPA/src/sample.py  --config ADJEPA/configs/sample_few_shot.yaml
```

### Model Training

Train on **GPU 0** using `ADJEPA/configs/train.yaml`:

```bash
python ADJEPA/main.py \
    --fname ADJEPA/configs/train.yaml \
    --mode train
```

### Anomaly Detection / Inference

After training, run inference:

```bash
python ADJEPA/main.py \
    --fname ADJEPA/configs/test.yaml \
    --mode AD
```



---

