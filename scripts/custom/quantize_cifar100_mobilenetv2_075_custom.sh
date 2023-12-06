#!/bin/sh
cd ../../

#src="/Users/tsgong/git/ai8x-training/logs/2023.11.14-212010/qat_checkpoint.pth.tar"
#src="/Users/tsgong/git/ai8x-training/logs/2023.11.15-012114/qat_checkpoint.pth.tar" # depthwise group-1, w/ res
#src="/Users/tsgong/git/ai8x-training/logs/2023.11.16-224332/qat_checkpoint.pth.tar" # fakedepthwise group-1, w/ res
#src="/Users/tsgong/git/ai8x-training/logs/2023.11.17-015308/qat_checkpoint.pth.tar" # remove depthwise, w/ res
#src="/Users/tsgong/git/ai8x-training/logs/2023.11.17-033321/qat_checkpoint.pth.tar" # remove depthwise, w/ res, bn 720 -> 480
#src="/Users/tsgong/git/ai8x-training/logs/2023.11.17-111801/qat_checkpoint.pth.tar" # remove depthwise, w/ res, bn 720 -> 480, 240->160
#src="/Users/tsgong/git/ai8x-training/logs/2023.11.17-111837/qat_checkpoint.pth.tar" # remove depthwise, w/ res, bn 720 -> 480, 240->160, 960 -> 640
#src="/Users/tsgong/git/ai8x-training/logs/2023.11.17-212924/qat_checkpoint.pth.tar" # remove depthwise, w/ res, bn 720 -> 480, 240->160, 960 -> 480
src="/Users/tsgong/git/ai8x-training/logs/2023.11.17-213143/qat_checkpoint.pth.tar" # remove depthwise, w/ res, bn 720 -> 480, 960 -> 480




dest="trained/custom/ai85-cifar100-mobilenet-v2-0.75-qat8-q_custom.pth.tar"
device="MAX78000"
config="networks/ai87-cifar100-mobilenet-v2-0.75.yaml"
python quantize.py $src $dest --device $device -v -c $config "$@"
