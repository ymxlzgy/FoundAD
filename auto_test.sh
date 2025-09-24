#!/bin/bash
diy_name="${1:-}"
start_step="${2:-}"
end_step="${3:-}"
interval="${4:-}"
master_port="${5:-}"
dataset="${6:-mvtec}"
dataset_name="${7:-mvtec_1shot}"
test_root="${8:-/media/ymxlzgy/Data21/xinyan/mvtec}"

echo "dataset=$dataset dataset_name=$dataset_name diy_name=$diy_name test_root=$test_root"
echo "steps: $start_step..$end_step step $interval"

for step in $(seq "$start_step" "$interval" "$end_step"); do

    python foundad/main.py \
        mode=AD \
        data.dataset="$dataset" \
        data.data_name="$dataset_name" \
        diy_name="$diy_name" \
        data.test_root="$test_root" \
        app=test \
        app.ckpt_step=$step \
        dist.master_port=$master_port
done