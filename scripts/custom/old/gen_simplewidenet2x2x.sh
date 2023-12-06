
mkdir raw_logs
#for SPLIT in 19; do
#cp="/home/tsgong/git/ai8x-training/logs/2023.10.30-015446/qat_checkpoint.pth.tar"
cp="/Users/tsgong/git/ai8x-training/logs/2023.10.30-015446/qat_checkpoint.pth.tar"
config="networks/custom/cifar100-simplewide2x2x.yaml"
device="MAX78000"

### full ###
#device="MAX78002"
#python ai8xize.py --verbose --test-dir exp --prefix cifar100-simplewide2x2x --checkpoint-file $CP_PATH --config-file $config --device $device --compact-data --stop-after ${SPLIT} --weight-start 0 --timer 0 --display-checkpoint --debug-latency --input-offset 0x0000 --overwrite

### LbLTT ###
#cd ../LbLTT
#python LbLTT.py configuration_examples/configuration-CIFAR100_simplenetwide2x2x.json
#cd ../ai8x-synthesis



for SPLIT in 16 17 18 19 20 21 22 23 24 25 26; do
#1st
  python ai8xize.py --verbose --test-dir exp --prefix cifar100-simplewide2x2x_0_$((SPLIT-1)) --checkpoint-file $CP_PATH --config-file $config --device $device --compact-data --stop-after ${SPLIT} --weight-start 0 --timer 0 --display-checkpoint --debug-latency --input-offset 0x0000 --overwrite 2>&1 | tee raw_logs/cifar100-simplewide2x2x_0_$((SPLIT-1)).txt &
#2nd

#  if [ `expr $SPLIT % 2` == 0 ]
#  then #even number
    OFFSET=0x0000
#  else #odd number
#    OFFSET=0x2000
#  fi

  python ai8xize.py --verbose --test-dir exp --prefix cifar100-simplewide2x2x_${SPLIT}_26 --checkpoint-file $CP_PATH --config-file $config --device $device --compact-data --weight-start 0 --input-offset ${OFFSET} --overwrite --skip-yaml-layers ${SPLIT} --skip-checkpoint-layers ${SPLIT} --timer 0 --display-checkpoint --debug-latency --sample-input ~/git/LbLTT/backup/input-samples-sipmlewidenet2x_transient/LAYER_${SPLIT}.npy 2>&1 | tee raw_logs/cifar100-simplewide2x2x_${SPLIT}_26.txt &
done


wait
echo "DONE!!!"