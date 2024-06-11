#!/bin/sh

cd ../../../

#input_shape="3x32x32"
input_shape="12x32x32"
#input_shape="48x32x32"

if [ "$input_shape" = "3x32x32" ]; then
  log_path="2024.04.23-154948"
  config="networks/cifar100-ressimplenet.yaml"
elif [ "$input_shape" = "12x32x32" ]; then
  log_path="2024.04.23-154955"
  config="networks/custom/data_folding/cifar100-ressimplenet_12.yaml"
elif [ "$input_shape" = "48x32x32" ]; then
  log_path="2024.04.23-183239"
  config="networks/custom/data_folding/cifar100-ressimplenet_48.yaml"
fi

############quantize
python quantize.py trained/custom/data_folding/$log_path/qat_checkpoint.pth.tar trained/custom/data_folding/$log_path/qat_checkpoint-q.pth.tar --device MAX78000 -v

dir='exp'
prefix="ai85-ressimplenet_${input_shape}"
cp="trained/custom/data_folding/${log_path}/qat_checkpoint-q.pth.tar"
device="MAX78000"
input_dir=${dir}/${prefix}_full/sample_inputs/LAYER

sp_st=0
sp_end=100

# generate sample inputs
python ai8xize.py --verbose --test-dir exp --prefix ${prefix}_full --checkpoint-file $cp --config-file $config --device $device --compact-data --weight-start 0 --input-offset 0x0000 --overwrite --timer 0 --display-checkpoint --debug-latency --save_input_samples --sample-input $input_shape


# split
python ai8xize.py --verbose --test-dir $dir --prefix ${prefix}_${sp_st}_${sp_end} --checkpoint-file $cp --config-file $config --device $device --compact-data --weight-start 0 --overwrite --timer 0 --display-checkpoint --debug-latency --skip-yaml-layers $sp_st --skip-checkpoint-layers $sp_st --stop-after $sp_end --sample-input ${input_dir}${sp_st}.npy --no-unload --overlap-data --no-kat

