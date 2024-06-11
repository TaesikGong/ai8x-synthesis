#!/bin/sh
cd ../../../

log_path="2024.02.26-163117"

python quantize.py trained/custom/data_folding/$log_path/qat_checkpoint.pth.tar trained/custom/data_folding/$log_path/qat_checkpoint-q.pth.tar --device MAX78000 -v
python izer/add_fake_passthrough.py --input-checkpoint-path trained/custom/data_folding/$log_path/qat_checkpoint-q.pth.tar --output-checkpoint-path trained/custom/data_folding/$log_path/qat_checkpoint-fakept-q.pth.tar --layer-name pt --layer-depth 56 --layer-name-after-pt upconv3
