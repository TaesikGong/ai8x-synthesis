

cd ../../
dir='exp'
prefix='ai85-cifar100-effnet2-custom-q'
cp="trained/custom/ai85-cifar100-effnet2_custom.pth.tar"
config="networks/custom/ai85-cifar100-effnet2_custom.yaml"
device="MAX78000"
input_dir=${dir}/${prefix}_full/sample_inputs/LAYER

sp_st=0
sp_end=20


#sp_st=20
#sp_end=30

# split
python ai8xize.py --verbose --test-dir $dir --prefix ${prefix}_${sp_st}_${sp_end} --checkpoint-file $cp --config-file $config --device $device --compact-data --weight-start 0 --overwrite --timer 0 --display-checkpoint --debug-latency --skip-yaml-layers $sp_st --skip-checkpoint-layers $sp_st --stop-after $sp_end --sample-input ${input_dir}${sp_st}.npy --input-offset 0x0000

# generate sample inputs
#python ai8xize.py --verbose --test-dir exp --prefix ${prefix}_full --checkpoint-file $cp --config-file $config --device $device --compact-data --weight-start 0 --input-offset 0x0000 --overwrite --timer 0 --display-checkpoint --debug-latency --save_input_samples




#### 87
#python ai8xize.py --verbose --test-dir exp --prefix ai87-cifar100-effnet2-reprod --checkpoint-file trained/custom/ai85-cifar100-effnet2_custom.pth.tar --config-file networks/ai87-cifar100-effnet2.yaml --device MAX78002 --compact-data --input-offset 0x0000 --overwrite --timer 0 --display-checkpoint --debug-latency
#
#
#python ai8xize.py --verbose --test-dir exp --prefix ai87-cifar100-effnet2-origin --checkpoint-file trained/ai87-cifar100-effnet2-qat8-q.pth.tar --config-file networks/ai87-cifar100-effnet2.yaml --device MAX78002 --compact-data --input-offset 0x0000 --overwrite --timer 0 --display-checkpoint --debug-latency

#quantized (nores, dim changed)
#python ai8xize.py --verbose --test-dir exp --prefix ai85-cifar100-effnet2-custom-q --checkpoint-file trained/custom_models/ai85_cifar100-q.pth.tar --config-file networks/custom/ai85-cifar100-effnet2_custom.yaml --device MAX78002 --compact-data --input-offset 0x0000 --overwrite --timer 0 --display-checkpoint --debug-latency


# before quant
#python ai8xize.py --verbose --test-dir exp --prefix ai85-cifar100-effnet2-custom-q --checkpoint-file /home/tsgong/git/ai8x-training/logs/2023.11.09-090619/qat_best.pth.tar --config-file networks/custom/ai85-cifar100-effnet2-nores_custom.yaml --device MAX78002 --compact-data --input-offset 0x0000 --overwrite --mlator