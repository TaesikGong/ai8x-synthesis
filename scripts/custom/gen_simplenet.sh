#!/bin/sh
cd ../../


dir='exp'
prefix='simplenet'
cp="trained/ai85-cifar100-qat-mixed-q.pth.tar"
config="networks/cifar100-simple.yaml"
device="MAX78000"
input_dir=${dir}/${prefix}_full/sample_inputs/LAYER

#sp_st=0
#sp_end=6


sp_st=6
sp_end=14

# split
python ai8xize.py --verbose --test-dir $dir --prefix ${prefix}_${sp_st}_${sp_end} --checkpoint-file $cp --config-file $config --device $device --compact-data --weight-start 0 --overwrite --timer 0 --display-checkpoint --debug-latency --skip-yaml-layers $sp_st --skip-checkpoint-layers $sp_st --stop-after $sp_end --sample-input ${input_dir}${sp_st}.npy --input-offset 0x0000

# generate sample inputs
#python ai8xize.py --verbose --test-dir exp --prefix ${prefix}_full --checkpoint-file $cp --config-file $config --device $device --compact-data --weight-start 0 --overwrite --timer 0 --display-checkpoint --debug-latency --save_input_samples


