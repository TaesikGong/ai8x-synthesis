cd ../../


dir='exp'
prefix='mnist'
cp="trained/ai85-mnist-qat8-q.pth.tar"
config="networks/mnist-chw-ai85.yaml"
device="MAX78000"
input_dir=${dir}/${prefix}_full/sample_inputs/LAYER

sp_st=0
sp_end=5

# split
echo python ai8xize.py --verbose --test-dir $dir --prefix ${prefix}_${sp_st}_${sp_end} --checkpoint-file $cp --config-file $config --device $device --compact-data --weight-start 0 --overwrite --timer 0 --skip-yaml-layers $sp_st --skip-checkpoint-layers $sp_st --display-checkpoint --debug-latency --stop-after $sp_end --input-offset 0x0000

# generate sample inputs
#python ai8xize.py --verbose --test-dir exp --prefix ${prefix}_full --checkpoint-file $cp --config-file $config --device $device --compact-data --weight-start 0 --overwrite --timer 0 --display-checkpoint --debug-latency --save_input_samples
