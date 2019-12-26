CUDA_VISIBLE_DEVICES=3 python -W ignore examples/run_glue.py     \
  --data_dir data/cola/                   \
  --model_type bert                       \
  --model_name_or_path bert-base-uncased  \
  --task_name cola                        \
  --output_dir result/cola/               \
  --evaluate_during_training --do_lower_case    \
  --do_train                              \
