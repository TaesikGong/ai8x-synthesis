#!/bin/sh

cd ../../../

input_shape="5x32x32"

#input_shape="3x32x32"
#input_shape="6x32x32"
#input_shape="18x32x32"
#input_shape="36x32x32"
#input_shape="64x32x32"

#log_path="2024.04.23-154954" #12
#log_path="2024.04.23-154956" #48

#PS3
#./240512-214207_Caltech101_simplenet_5_coordconv_000000000000000_1.txt ; /home/taesik/git/ai8x-training/logs/2024.05.12-214239/2024.05.12-214239.log

#./240503-230934_Caltech101_simplenet_3_000000000000000.txt ; /home/taesik/git/ai8x-training/logs/2024.05.03-230945/2024.05.03-230945.log
#./240509-095544_Caltech101_simplenet_6_000000000000000_1.txt ; /home/taesik/git/ai8x-training/logs/2024.05.09-095556/2024.05.09-095556.log
#./240509-095550_Caltech101_simplenet_36_000000000000000_1.txt ; /home/taesik/git/ai8x-training/logs/2024.05.09-095602/2024.05.09-095602.log
#./240503-230946_Caltech101_simplenet_64_000000000000000.txt ; /home/taesik/git/ai8x-training/logs/2024.05.03-230959/2024.05.03-230959.log

#PS2
#./240504-165351_Caltech101_simplenet_18_000000000000000_1.txt; /home/taesik/git/ai8x-training/logs/2024.05.04-165410/2024.05.04-165410.log

if [ "$input_shape" = "3x32x32" ]; then
  log_path="2024.05.03-230945"
  config="networks/cifar100-simple.yaml"
elif [ "$input_shape" = "5x32x32" ]; then
  log_path="2024.05.12-214239"
  config="networks/custom/data_folding/cifar100-simple_5.yaml"
elif [ "$input_shape" = "6x32x32" ]; then
  log_path="2024.05.09-095556"
  config="networks/custom/data_folding/cifar100-simple_6.yaml"
elif [ "$input_shape" = "18x32x32" ]; then
  log_path="2024.05.04-165410"
  config="networks/custom/data_folding/cifar100-simple_18.yaml"
elif [ "$input_shape" = "36x32x32" ]; then
  log_path="2024.05.09-095602"
  config="networks/custom/data_folding/cifar100-simple_36.yaml"
elif [ "$input_shape" = "64x32x32" ]; then
  log_path="2024.05.03-230959"
  config="networks/custom/data_folding/cifar100-simple_64.yaml"
fi

############quantize
python quantize.py trained/custom/data_folding/$log_path/qat_checkpoint.pth.tar trained/custom/data_folding/$log_path/checkpoint-q.pth.tar --device MAX78000 -v

dir='exp'
prefix="ai85-simplenet_${input_shape}"
cp="trained/custom/data_folding/${log_path}/checkpoint-q.pth.tar"
device="MAX78000"
input_dir=${dir}/${prefix}_full/sample_inputs/LAYER

sp_st=0
sp_end=100

# generate sample inputs
python ai8xize.py --verbose --test-dir exp --prefix ${prefix}_full --checkpoint-file $cp --config-file $config --device $device --compact-data --weight-start 0 --input-offset 0x0000 --overwrite --timer 0 --display-checkpoint --debug-latency --save_input_samples --sample-input $input_shape


# split
python ai8xize.py --verbose --test-dir $dir --prefix ${prefix}_${sp_st}_${sp_end} --checkpoint-file $cp --config-file $config --device $device --compact-data --weight-start 0 --overwrite --timer 0 --display-checkpoint --debug-latency --skip-yaml-layers $sp_st --skip-checkpoint-layers $sp_st --stop-after $sp_end --sample-input ${input_dir}${sp_st}.npy --no-unload --overlap-data --no-kat

