CUDA_VISIBLE_DEVICES=3 python -W ignore examples/run_glue.py     \
  --data_dir data/rte/                    \
  --model_type bert                       \
  --model_name_or_path bert-base-uncased  \
  --task_name rte                         \
  --output_dir result/rte/                \
  --evaluate_during_training --do_lower_case    \
  --quantify                              \
  --num_train_epochs 9
