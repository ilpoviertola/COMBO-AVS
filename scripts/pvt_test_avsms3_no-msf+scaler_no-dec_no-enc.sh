dataset_root=${2:-'AVS_dataset/AVSBench_object/Multi-sources/'}
export DETECTRON2_DATASETS=$dataset_root
export CUDA_VISIBLE_DEVICES=0
python pred.py \
    --num-gpus 1 \
    --config-file configs/avs_ms3/COMBO_PVTV2B5_bs4_20k_no-msf+scaler+_no-dec_no-enc.yaml \
    --dist-url tcp://0.0.0.0:47773 \
    --eval-only \
