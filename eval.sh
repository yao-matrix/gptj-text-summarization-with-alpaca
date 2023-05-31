torchrun --nproc_per_node=8 --master_port=23461 eval.py \
    --model_name_or_path ./model/checkpoint-52000 \
    --data_path ./data/cnn_eval.json \
