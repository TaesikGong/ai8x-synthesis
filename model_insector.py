import torch


original_fold4='trained/ai85-camvid-unet-large.pth.tar'
fold4 = 'trained/custom/data_folding/2024.02.26-163018/checkpoint.pth.tar'
fold1 = 'trained/custom/data_folding/2024.02.26-163117/checkpoint.pth.tar'
qat_fold4 = 'trained/custom/data_folding/2024.02.26-163018/qat_checkpoint.pth.tar'
qat_fold1 = 'trained/custom/data_folding/2024.02.26-163117/qat_checkpoint.pth.tar'
fold4_model = torch.load(fold4, map_location=torch.device('cpu'))
fold1_model = torch.load(fold1, map_location=torch.device('cpu'))
qat_fold4_model = torch.load(qat_fold4, map_location=torch.device('cpu'))
qat_fold1_model = torch.load(qat_fold1, map_location=torch.device('cpu'))
original_fold4_model = torch.load(original_fold4, map_location=torch.device('cpu'))
print(fold4_model)