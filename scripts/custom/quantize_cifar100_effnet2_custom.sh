#!/bin/sh

cd ../../
#src="/Users/tsgong/git/ai8x-synthesis/trained/ai87-cifar100-effnet2-qat8.pth.tar"
#src="/Users/tsgong/git/ai8x-training/logs/2023.11.14-210645/qat_checkpoint.pth.tar" # w/o depthwise, w/ res
#src="/Users/tsgong/git/ai8x-training/logs/2023.11.15-012837/qat_checkpoint.pth.tar" # depthwise group-1, w/ res
#src="/Users/tsgong/git/ai8x-training/logs/2023.11.16-214027/qat_checkpoint.pth.tar" # fakedepthwise group-1, w/ res
#src="/Users/tsgong/git/ai8x-training/logs/2023.11.17-015247/qat_checkpoint.pth.tar" # remove depthwise, w/ res
src="/Users/tsgong/git/ai8x-training/logs/2023.11.17-032749/qat_checkpoint.pth.tar" # remove depthwise, w/ res, bn 1024 -> 512


dest='trained/custom/ai85-cifar100-effnet2_custom.pth.tar'
config="networks/custom/ai85-cifar100-effnet2_custom.yaml"
device="MAX78000"
#config="networks/ai87-cifar100-effnet2.yaml"
python quantize.py $src $dest --device $device -v -c $config "$@"

#python quantize.py trained/ai87-cifar100-effnet2-qat8.pth.tar trained/ai87-cifar100-effnet2-qat8-q.pth.tar --device MAX78002 -v -c networks/ai87-cifar100-effnet2.yaml "$@"

# (nores, dim changed)
#python quantize.py /home/tsgong/git/ai8x-training/logs/2023.11.09-090619/qat_best.pth.tar trained/custom_models/ai85_cifar100-q.pth.tar --device MAX78000 -v -c networks/custom/ai85-cifar100-effnet2_custom.yaml "$@"

