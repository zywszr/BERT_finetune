CUDA_VISIBLE_DEVICES=2 python -W ignore examples/run_glue.py     \
  --data_dir data/wnli/                   \
  --model_type bert                       \
  --model_name_or_path bert-base-uncased  \
  --task_name wnli                        \
  --output_dir result/wnli/               \
  --evaluate_during_training --do_lower_case    \
  --quantify                              \
  --num_train_epochs 30
