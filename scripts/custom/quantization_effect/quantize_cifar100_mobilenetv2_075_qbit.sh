#!/bin/sh
cd ../../../

src="trained/custom/quantization_effect/2024.02.17-104953/qat_best.pth.tar" # 8bit
dest="trained/custom/quantization_effect/ai87-cifar100-mobilenet-v2-0.75-qat8-q.pth.tar"

#src="trained/custom/quantization_effect/2024.02.17-105230/qat_best.pth.tar" # 4bit
#dest="trained/custom/quantization_effect/ai87-cifar100-mobilenet-v2-0.75-qat4-q.pth.tar"
#
#src="trained/custom/quantization_effect/2024.02.17-105231/qat_best.pth.tar" # 2bit
#dest="trained/custom/quantization_effect/ai87-cifar100-mobilenet-v2-0.75-qat2-q.pth.tar"
#
#src="trained/custom/quantization_effect/2024.02.17-105234/qat_best.pth.tar" # 1bit
#dest="trained/custom/quantization_effect/ai87-cifar100-mobilenet-v2-0.75-qat1-q.pth.tar"


device="MAX78002"
config="networks/ai87-cifar100-mobilenet-v2-0.75.yaml"
python quantize.py $src $dest --device $device -v -c $config "$@"
