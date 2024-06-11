

cd ../../../
dir='exp'
prefix="ai85-camvid-unet-large-q_fold4"
cp="trained/custom/data_folding/2024.02.26-163018/qat_checkpoint-q.pth.tar"
config="networks/custom/camvid-unet-large_fold4.yaml"
device="MAX78000"
input_dir=${dir}/${prefix}_full/sample_inputs/LAYER


sp_st=0
sp_end=100

# split
python ai8xize.py --verbose --test-dir $dir --prefix ${prefix}_${sp_st}_${sp_end} --checkpoint-file $cp --config-file $config --device $device --compact-data --weight-start 0 --overwrite --timer 0 --display-checkpoint --debug-latency --skip-yaml-layers $sp_st --skip-checkpoint-layers $sp_st --stop-after $sp_end --sample-input ${input_dir}${sp_st}.npy --no-unload --overlap-data --no-kat

# generate sample inputs
input_shape="48x88x88" # "sample_camvid_s352_c3_reduced.npy" has 48x48x48.... due to memory size..
#python ai8xize.py --verbose --test-dir exp --prefix ${prefix}_full --checkpoint-file $cp --config-file $config --device $device --compact-data --weight-start 0 --input-offset 0x0000 --overwrite --timer 0 --display-checkpoint --debug-latency --save_input_samples --sample-input $input_shape
