#!/bin/sh
cd ../../../

dir='exp'
config="networks/ai87-cifar100-mobilenet-v2-0.75.yaml"
device="MAX78002"

#for q in 8 4 2 1; do
#    cp="trained/custom/quantization_effect/ai87-cifar100-mobilenet-v2-0.75-qat${q}-q.pth.tar"
#    prefix="mobilenetv2-075-qat${q}"
#
#    python ai8xize.py --verbose --test-dir ${dir} --prefix ${prefix} --checkpoint-file ${cp} --config-file ${config} --device ${device} --compact-data --weight-start 0 --overwrite --timer 0 --display-checkpoint --debug-latency --input-offset 0x0000
#done



cp="trained/custom/quantization_effect/2024.02.19-160949/best.pth.tar"
prefix="mobilenetv2-075-qat-no"

python ai8xize.py --verbose --test-dir ${dir} --prefix ${prefix} --checkpoint-file ${cp} --config-file ${config} --device ${device} --compact-data --weight-start 0 --overwrite --timer 0 --display-checkpoint --debug-latency --input-offset 0x0000