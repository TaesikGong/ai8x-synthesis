

cd ../../
dir='exp'
prefix="ai85-camvid-unet-large-q"
cp="trained/ai85-camvid-unet-large-q.pth.tar"
config="networks/custom/camvid-unet-large.yaml"
device="MAX78000"
input_dir=${dir}/${prefix}_full/sample_inputs/LAYER


sp_st=15
sp_end=19

# split
python ai8xize.py --verbose --test-dir $dir --prefix ${prefix}_${sp_st}_${sp_end} --checkpoint-file $cp --config-file $config --device $device --compact-data --weight-start 0 --overwrite --timer 0 --display-checkpoint --debug-latency --skip-yaml-layers $sp_st --skip-checkpoint-layers $sp_st --stop-after $sp_end --sample-input ${input_dir}${sp_st}.npy --overwrite --no-unload --overlap-data --no-kat

# generate sample inputs
#python ai8xize.py --verbose --test-dir exp --prefix ${prefix}_full --checkpoint-file $cp --config-file $config --device $device --compact-data --weight-start 0 --input-offset 0x0000 --overwrite --timer 0 --display-checkpoint --debug-latency --save_input_samples




############################camvid_unet_fakept##############################
##### Same as /Users/tsgong/MaximSDK/Examples/MAX78000/CNN/camvid_unet_fakept, but removed "--mlator" option. #########

##### Note that `--no-unload` is required due to SRAM issue
##### Note that `--max-checklines 0`  is required due to SRAM issue
##### Note that `--overlap-data` is required to avoid memory overlapping issue in YAML.

#python ai8xize.py --test-dir exp --prefix camvid_unet_fakept --checkpoint-file trained/ai85-camvid-unet-large-fakept-q.pth.tar --config-file networks/camvid-unet-large-fakept.yaml --device MAX78000 --timer 0 --display-checkpoint --verbose --overlap-data --max-checklines 8192 --new-kernel-loader --overwrite --debug-latency --no-unload
