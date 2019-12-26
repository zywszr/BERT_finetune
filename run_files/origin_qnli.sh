CUDA_VISIBLE_DEVICES=3 python -W ignore examples/run_glue.py     \
  --data_dir data/qnli/                   \
  --model_type bert                       \
  --model_name_or_path bert-base-uncased  \
  --task_name qnli                        \
  --output_dir result/qnli/               \
  --evaluate_during_training --do_lower_case    \
  --do_train                              \
  --num_train_epochs 1
