# cd /project/directory

CUDA_VISIBLE_DEVICES=2 python src/test/test_CAMELYON16.py \
  --model_type  \
  --model_path ./weights/PathSearch_best_validation.pt \
  --data_dir ./data/DHMC__LUAD/ \
  --feature_method model \
  --batch_size 1 \
  --sample_num -1 \
  --device cpu