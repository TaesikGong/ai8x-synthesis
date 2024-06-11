

cd ../../

dir='exp'
prefix='ai85-cifar100-mobilenet-v2-075-q_custom'
cp="trained/custom/ai85-cifar100-mobilenet-v2-0.75-qat8-q_custom.pth.tar"
#config="networks/ai87-cifar100-mobilenet-v2-0.75.yaml"
config="networks/custom/ai85-cifar100-mobilenet-v2-0.75_custom.yaml"
device="MAX78000"
input_dir=${dir}/${prefix}_full/sample_inputs/LAYER

sp_st=0
sp_end=21

#sp_st=32
#sp_end=42

#sp_st=42
#sp_end=52

#sp_st=52
#sp_end=56

############# debug #############
#sp_st=33
#sp_end=35
###/Users/tsgong/MaximSDK/Tools/GNUTools/10.3/bin/../lib/gcc/arm-none-eabi/10.3.1/../../../../arm-none-eabi/bin/ld: region `SRAM' overflowed by 316000 bytes
   # no SARM overlfow without cnn_unload.
   # changed CNN_NUM_OUTPUTS to 120000 and modified the cnn_unload function (commented out). it works!

#sp_st=43
#sp_end=45
###/Users/tsgong/MaximSDK/Tools/GNUTools/10.3/bin/../lib/gcc/arm-none-eabi/10.3.1/../../../../arm-none-eabi/bin/ld: region `FLASH' overflowed by 55920 bytes
   #/Users/tsgong/MaximSDK/Tools/GNUTools/10.3/bin/../lib/gcc/arm-none-eabi/10.3.1/../../../../arm-none-eabi/bin/ld: region `SRAM' overflowed by 365152 bytes
   # no SARM overlfow without cnn_unload

#32-45 shows error: ERROR: Layer 0 (l42): bias memory capacity exhausted - available groups: [0, 1, 2, 3], used so far: [480, 432, 432, 432], needed: 288 bytes, best available: group 1 with 80 bytes available.

# split
python ai8xize.py --verbose --test-dir $dir --prefix ${prefix}_${sp_st}_${sp_end} --checkpoint-file $cp --config-file $config --device $device --compact-data --weight-start 0 --overwrite --timer 0 --display-checkpoint --debug-latency --skip-yaml-layers $sp_st --skip-checkpoint-layers $sp_st --stop-after $sp_end --sample-input ${input_dir}${sp_st}.npy --input-offset 0x0000 --overlap-data --no-kat

# generate sample inputs
#python ai8xize.py --verbose --test-dir exp --prefix ${prefix}_full --checkpoint-file $cp --config-file $config --device $device --compact-data --weight-start 0 --input-offset 0x0000 --overwrite --timer 0 --display-checkpoint --debug-latency --save_input_samples
