
CUDA_VISIBLE_DEVICES=1 python finetune_on_pregenerated.py \
    --pregenerated_data ../../../../workspace/CBert/training/ \
    --bert_model bert-base-uncased \
    --do_lower_case \
    --output_dir finetuned_lm/ \
    --epochs 3 \
    --debug \
    --no_cuda     

