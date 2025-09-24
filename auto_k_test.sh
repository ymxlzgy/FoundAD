#!/bin/bash
diy_name="${1:-}"
start_k="${2:-}"
end_k="${3:-}"
interval="${4:-}"
step="${5:-}"
master_port="${6:-}"
dataset="${7:-mvtec}"
dataset_name="${8:-mvtec_1shot}"
test_root="${9:-/media/ymxlzgy/Data21/xinyan/mvtec}"

echo "dataset=$dataset dataset_name=$dataset_name diy_name=$diy_name test_root=$test_root"
echo "K: $start_k..$end_k skip $interval"

for k in $(seq "$start_k" "$interval" "$end_k"); do

    python foundad/main.py \
        mode=AD \
        data.dataset="$dataset" \
        data.data_name="$dataset_name" \
        diy_name="$diy_name" \
        data.test_root="$test_root" \
        app=test \
        app.ckpt_step=$step \
        testing.K_top_mvtec=$k \
        testing.K_top_visa=$k \
        dist.master_port=$master_port
done