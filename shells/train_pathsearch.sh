# cd /project/directory

CUDA_VISIBLE_DEVICES=0,1 python src/train_pathsearch.py \
--batch_size 256 \
--learning_rate 8e-5 \
--sample_num 512 \
--img_sup_loss_weight 0.3 \
--text_sup_loss_weight 0.3 \
--save_dir ./weights/ > ./train_pathsearch.log 2>&1 &
