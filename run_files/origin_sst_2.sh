CUDA_VISIBLE_DEVICES=1 python -W ignore examples/run_glue.py     \
  --data_dir data/sst_2/                  \
  --model_type bert                       \
  --model_name_or_path bert-base-uncased  \
  --task_name sst-2                       \
  --output_dir result/sst_2/              \
  --evaluate_during_training --do_lower_case    \
  --do_train                              \
  --num_train_epochs 1
