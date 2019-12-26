
python pregenerate_training_data.py \
        --train_corpus raw_data.txt \
        --bert_model bert-base-uncased \
        --do_lower_case \
        --output_dir training/ \
        --epochs_to_generate 3 \
        --max_seq_len 256
