#!/bin/sh
cd ../../../

#src="trained/custom/quantization_effect/2024.02.17-110119/qat_best.pth.tar" # 8bit
#dest="trained/custom/quantization_effect/ai87-cifar100-effnet2-qat8-q.pth.tar"

#src="trained/custom/quantization_effect/2024.02.17-110132/qat_best.pth.tar" # 4bit
#dest="trained/custom/quantization_effect/ai87-cifar100-effnet2-qat4-q.pth.tar"
#
#src="trained/custom/quantization_effect/2024.02.17-110158/qat_best.pth.tar" # 2bit
#dest="trained/custom/quantization_effect/ai87-cifar100-effnet2-qat2-q.pth.tar"
#
#src="trained/custom/quantization_effect/2024.02.17-110226/qat_best.pth.tar" # 1bit
#dest="trained/custom/quantization_effect/ai87-cifar100-effnet2-qat1-q.pth.tar"


device="MAX78002"
config="networks/ai87-cifar100-effnet2.yaml"
python quantize.py $src $dest --device $device -v -c $config "$@"
