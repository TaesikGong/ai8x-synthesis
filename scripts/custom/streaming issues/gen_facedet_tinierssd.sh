#!/bin/sh
cd ../../


dir='exp'
prefix='facedet_tinierssd'
cp="trained/ai85-facedet-tinierssd-qat8-q.pth.tar"
config="networks/ai87-facedet-tinierssd.yaml" # ai85 model has streaming, which is not our focus. ai87 model doesn't have.
device="MAX78000"
input_dir=${dir}/${prefix}_full/sample_inputs/LAYER

sp_st=0
sp_end=5

#### ERROR: ERROR: Layer 0: 1 channel/word 224x168 input (size 37632) with input offset 0x0000 and expansion 1x exceeds data memory instance size of 32768.

# split
python ai8xize.py --verbose --test-dir $dir --prefix ${prefix}_${sp_st}_${sp_end} --checkpoint-file $cp --config-file $config --device $device --compact-data --weight-start 0 --overwrite --timer 0 --display-checkpoint --debug-latency --skip-yaml-layers $sp_st --skip-checkpoint-layers $sp_st --stop-after $sp_end --sample-input ${input_dir}${sp_st}.npy --input-offset 0x0000

# generate sample inputs
#python ai8xize.py --verbose --test-dir exp --prefix ${prefix}_full --checkpoint-file $cp --config-file $config --device $device --compact-data --weight-start 0 --overwrite --timer 0 --display-checkpoint --debug-latency --save_input_samples

