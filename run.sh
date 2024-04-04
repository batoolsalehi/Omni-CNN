#bin/bash

# when --cont is set, --epochs has to be higher than the number of epochs executed when the model has been saved the last time (this information is written and retrieved from the weights file name)
# ----------------------------------------------------------------------------------------------------
mkdir -p /home/batool/omni/experiments/$1;
CUDA_VISIBLE_DEVICES=$2 python main.py --dataset flash --exp_name flash --base_path /home/batool/omni/flash_GPS_Image_LiDAR/ --save_path /home/batool/omni/experiments/$1/ --load-model '' --load-model-pruned '' --classes 64  --sparsity-type irregular --epochs 150 --epochs-prune 150 --epochs-mask-retrain 150 --admm-epochs 3 --mask-admm-epochs 10  --rho 0.01  --mixup --alpha 0 --smooth --smooth-eps 0.1 --config-setting 3,5,2 --adaptive-mask False --adaptive-ratio 0 --arch flashnet --depth 10 --tasks 3 \
> /home/batool/omni/experiments/$1/log.out \
2> /home/batool/omni/experiments/$1/log.err
