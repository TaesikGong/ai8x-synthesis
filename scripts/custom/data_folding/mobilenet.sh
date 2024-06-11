#!/bin/sh

cd ../../../

input_shape="5x32x32"

#input_shape="3x32x32"
#input_shape="6x32x32"
#input_shape="18x32x32"
#input_shape="36x32x32"
#input_shape="64x32x32"

#PS3
#./240512-233735_Caltech101_mobilenetv2_075_5_coordconv_000000000000000_1.txt; /home/taesik/git/ai8x-training/logs/2024.05.12-233817/2024.05.12-233817.log

#./240503-230939_Caltech101_mobilenetv2_075_3_000000000000000.txt ; /home/taesik/git/ai8x-training/logs/2024.05.03-230951/2024.05.03-230951.log
#./240506-101631_Caltech101_mobilenetv2_075_6_000000000000000_0.txt ; /home/taesik/git/ai8x-training/logs/2024.05.06-101642/2024.05.06-101642.log
#./240503-230945_Caltech101_mobilenetv2_075_18_000000000000000.txt ; /home/taesik/git/ai8x-training/logs/2024.05.03-230957/2024.05.03-230957.log
#./240506-101637_Caltech101_mobilenetv2_075_36_000000000000000_0.txt ; /home/taesik/git/ai8x-training/logs/2024.05.06-101651/2024.05.06-101651.log
#./240503-230951_Caltech101_mobilenetv2_075_64_000000000000000.txt ; /home/taesik/git/ai8x-training/logs/2024.05.03-231005/2024.05.03-231005.log

if [ "$input_shape" = "3x32x32" ]; then
  log_path="2024.05.03-230951"
  config="networks/ai87-cifar100-mobilenet-v2-0.75.yaml"
elif [ "$input_shape" = "5x32x32" ]; then
  log_path="2024.05.12-233817"
  config="networks/custom/data_folding/ai87-cifar100-mobilenet-v2-0.75_5.yaml"
elif [ "$input_shape" = "6x32x32" ]; then
  log_path="2024.05.06-101642"
  config="networks/custom/data_folding/ai87-cifar100-mobilenet-v2-0.75_6.yaml"
elif [ "$input_shape" = "18x32x32" ]; then
  log_path="2024.05.03-230957"
  config="networks/custom/data_folding/ai87-cifar100-mobilenet-v2-0.75_18.yaml"
elif [ "$input_shape" = "36x32x32" ]; then
  log_path="2024.05.06-101651"
  config="networks/custom/data_folding/ai87-cifar100-mobilenet-v2-0.75_36.yaml"
elif [ "$input_shape" = "64x32x32" ]; then
  log_path="2024.05.03-231005"
  config="networks/custom/data_folding/ai87-cifar100-mobilenet-v2-0.75_64.yaml"
fi

############quantize
python quantize.py trained/custom/data_folding/$log_path/qat_checkpoint.pth.tar trained/custom/data_folding/$log_path/checkpoint-q.pth.tar --device MAX78002 -v -c $config


dir='exp'
prefix="ai87-mobilenet_${input_shape}"
cp="trained/custom/data_folding/${log_path}/checkpoint-q.pth.tar"
device="MAX78002"
input_dir=${dir}/${prefix}_full/sample_inputs/LAYER

sp_st=0
sp_end=100

# generate sample inputs
python ai8xize.py --verbose --test-dir exp --prefix ${prefix}_full --checkpoint-file $cp --config-file $config --device $device --compact-data --weight-start 0 --input-offset 0x0000 --overwrite --timer 0 --display-checkpoint --debug-latency --save_input_samples --sample-input $input_shape


# split
python ai8xize.py --verbose --test-dir $dir --prefix ${prefix}_${sp_st}_${sp_end} --checkpoint-file $cp --config-file $config --device $device --compact-data --weight-start 0 --overwrite --timer 0 --display-checkpoint --debug-latency --skip-yaml-layers $sp_st --skip-checkpoint-layers $sp_st --stop-after $sp_end --sample-input ${input_dir}${sp_st}.npy --no-unload --overlap-data --no-kat

