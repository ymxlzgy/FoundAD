# FoundAD

The implementation of the submission **Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors**.
   

## Environment Setup

All Python dependencies are listed in `requirements.txt`. We recommend Python ≥ 3.10.

```bash
conda create -n foundad python=3.10
conda activate foundad
cd FoundAD
pip install -r requirements.txt
pip install -e .
```

Before we start, please make sure you have the rights to access DINOv3.

## Training and Inference

### Dataset Preparation

Download MVTec AD from their webiste and VisA from RealNet.

### Few-Shot Sampling

Create a **few-shot** subset with `sample.py`:

```bash
python foundad/src/sample.py source=/path/to/visa target=/path/to/visa-2-shot seed=45 num_samples=2
```
where `source` is the dataset folder, `target` is the folder of few-shot samples, and `num_samples` is the number of samples training models, e.g., 2 for 2-shot learning. `seed` can be adjusted to have multiple rounds of experiment.

### Model Training

```bash
python foundad/main.py mode=train data.batch_size=8 data.dataset=mvtec data.data_name=mvtec_1shot data.data_path=/path/to/few-shot-dataset app=train_dinov3 diy_name=dbug data.use_rotate90=True data.use_vflip=True data.use_color_jitter=True data.use_gray=True data.use_blur=True dist.master_port=40114 optimization.lr_config=const optimization.lr=0.001 app.meta.feat_normed=False
```
where `data.dataset` is "mvtec" or "visa", `data.data_name` is the folder name of few-shot samples, `data.data_path` is the path where the few-shot folder is at, `app` is "train_dinov2" or "train_dinov3" under `configs/app/`, and `diy_name` (optionally) is the post-fix name of the model saving directory. To adjust the layer, please specify `app.meta.n_layer`.

### Inference

After training, run inference:

```bash
python foundad/main.py mode=AD data.dataset=mvtec data.data_name=mvtec_1shot diy_name=dbug data.test_root=/path/to/mvtec app=test app.ckpt_step=3600
```
where `data.test_root` is the dataset folder, and `app` is test_dinov2 or test_dinov3 under `configs/app/`. To adjust sample number K, please specify `testing.K_top_mvtec` and `testing.K_top_visa`.