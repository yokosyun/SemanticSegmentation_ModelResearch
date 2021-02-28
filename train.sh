python3 train.py --model unet \
                --data_root /media/yoko/SSD-PGU3/workspace/datasets/citysacepes \
                --train_path data/cityscapes/train.txt \
                --img_dir JPEGImages \
                --mask_dir SegmentationClass \
                --save_dir result \
                --batch_size 1 \
                --gpu 0