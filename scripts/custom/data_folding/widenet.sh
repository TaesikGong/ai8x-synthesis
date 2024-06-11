#!/bin/sh

cd ../../../

input_shape="5x32x32"

#input_shape="3x32x32"
#input_shape="6x32x32"
#input_shape="18x32x32"
#input_shape="36x32x32"
#input_shape="64x32x32"

## all from ps3
# ./240512-222545_Caltech101_widenet_5_coordconv_000000000000000_1.txt;  /home/taesik/git/ai8x-training/logs/2024.05.12-222626/2024.05.12-222626.log

#./240515-094706_Caltech101_widenet_3_data-reshape_000000000000000_1.txt ; /home/taesik/git/ai8x-training/logs/2024.05.15-094720/2024.05.15-094720.log
#./240515-094708_Caltech101_widenet_6_data-reshape_000000000000000_1.txt ; /home/taesik/git/ai8x-training/logs/2024.05.15-094721/2024.05.15-094721.log
#./240515-094709_Caltech101_widenet_18_data-reshape_000000000000000_1.txt ; /home/taesik/git/ai8x-training/logs/2024.05.15-094723/2024.05.15-094723.log
#./240515-094711_Caltech101_widenet_36_data-reshape_000000000000000_1.txt ; /home/taesik/git/ai8x-training/logs/2024.05.15-094725/2024.05.15-094725.log
#./240515-094712_Caltech101_widenet_64_data-reshape_000000000000000_1.txt ; /home/taesik/git/ai8x-training/logs/2024.05.15-094728/2024.05.15-094728.log



if [ "$input_shape" = "3x32x32" ]; then
  log_path="2024.05.15-094720"
  config="networks/cifar100-simplewide2x.yaml"
elif [ "$input_shape" = "5x32x32" ]; then
  log_path="2024.05.12-222626"
  config="networks/custom/data_folding/cifar100-simplewide2x_5.yaml"
elif [ "$input_shape" = "6x32x32" ]; then
  log_path="2024.05.15-094721"
  config="networks/custom/data_folding/cifar100-simplewide2x_6.yaml"
elif [ "$input_shape" = "18x32x32" ]; then
  log_path="2024.05.15-094723"
  config="networks/custom/data_folding/cifar100-simplewide2x_18.yaml"
elif [ "$input_shape" = "36x32x32" ]; then
  log_path="2024.05.15-094725"
  config="networks/custom/data_folding/cifar100-simplewide2x_36.yaml"
elif [ "$input_shape" = "64x32x32" ]; then
  log_path="2024.05.15-094728"
  config="networks/custom/data_folding/cifar100-simplewide2x_64.yaml"
fi

############quantize
python quantize.py trained/custom/data_folding/$log_path/qat_checkpoint.pth.tar trained/custom/data_folding/$log_path/checkpoint-q.pth.tar --device MAX78000 -v

dir='exp'
prefix="ai85-widenet_${input_shape}"
cp="trained/custom/data_folding/${log_path}/checkpoint-q.pth.tar"
device="MAX78000"
input_dir=${dir}/${prefix}_full/sample_inputs/LAYER

sp_st=0
sp_end=100

## generate sample inputs
python ai8xize.py --verbose --test-dir exp --prefix ${prefix}_full --checkpoint-file $cp --config-file $config --device $device --compact-data --weight-start 0 --input-offset 0x0000 --overwrite --timer 0 --display-checkpoint --debug-latency --save_input_samples --sample-input $input_shape
#
#
## split
python ai8xize.py --verbose --test-dir $dir --prefix ${prefix}_${sp_st}_${sp_end} --checkpoint-file $cp --config-file $config --device $device --compact-data --weight-start 0 --overwrite --timer 0 --display-checkpoint --debug-latency --skip-yaml-layers $sp_st --skip-checkpoint-layers $sp_st --stop-after $sp_end --sample-input ${input_dir}${sp_st}.npy --no-unload --overlap-data --no-kat
#
