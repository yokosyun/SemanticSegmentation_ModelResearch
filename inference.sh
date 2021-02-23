python3 inference.py --model unet \
                --data_root /media/yoko/SSD-PGU3/workspace/datasets/VOCdevkit/VOC2012 \
                --test_path ImageSets/Segmentation/trainval.txt \
                --img_dir JPEGImages \
                --mask_dir SegmentationClass \
                --save_dir result \
                --batch_size 1 \
                --checkpoint result/weight/unet_best.pth \
                --gpu 0