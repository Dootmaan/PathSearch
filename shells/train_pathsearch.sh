# cd /project/directory

# CUDA_VISIBLE_DEVICES=0,1 python src/train_pathsearch.py \
# --batch_size 256 \
# --learning_rate 8e-5 \
# --sample_num 512 \
# --img_sup_loss_weight 0.3 \
# --text_sup_loss_weight 0.3 \
# --save_dir ./weights/ > ./train_pathsearch.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,2 python src/train_pathsearch.py \
--data_root /your/path/to/DATA/VLP/wsi_report/vanilla_conch_v1_5/ \
--batch_size 12 \
--learning_rate 8e-5 \
--sample_num 512 \
--img_sup_loss_weight 0.3 \
--text_sup_loss_weight 0.3 \
--save_dir ./weights/ > ./train_pathsearch.log 2>&1 &
