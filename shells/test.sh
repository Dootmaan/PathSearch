# cd /path/to/PathSearch
# Run evaluation on all four public benchmarks used in the paper.
# Adjust --data_dir and --model_path for your local setup.

MODEL=./weights/PathSearch_best_validation.pt

# CAMELYON16
CUDA_VISIBLE_DEVICES=0 python src/test/test_CAMELYON16.py \
  --model_path $MODEL \
  --data_dir ./data/CAMELYON16 \
  --batch_size 1 \
  --sample_num -1

# CAMELYON17
CUDA_VISIBLE_DEVICES=0 python src/test/test_CAMELYON17.py \
  --model_path $MODEL \
  --data_dir ./data/CAMELYON17 \
  --batch_size 1 \
  --sample_num -1

# DHMC-LUAD
CUDA_VISIBLE_DEVICES=0 python src/test/test_DHMCLUAD.py \
  --model_path $MODEL \
  --data_dir ./data/DHMC_LUAD \
  --batch_size 1 \
  --sample_num -1

# TCGA
CUDA_VISIBLE_DEVICES=0 python src/test/test_TCGA.py \
  --model_path $MODEL \
  --data_dir ./data/TCGA \
  --sample_num -1
