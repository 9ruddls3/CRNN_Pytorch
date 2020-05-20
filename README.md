# CRNN_Pytorch
This package is for my custom Dataset HearthStone img) and based on Google Colaborative.   
If you use this for your custom dataset, you can convert your image folder to torchvision.Dataloader by using prepare.py


이 CRNN 패키지는 제 데이터셋을 기반으로 진행하였고, 구글 코랩을 기반으로 진행하였습니다.   
전반적인 Hyper-Parameter는 prams.py 에서 선언되었으며, 이를 수정함으로써 튜닝하는것도 방법입니다.   
문자열 이미지데이터가 있고, 파일명을 label.확장자로 되어있다면, prepare.py 를 통해 Datalodar 로 변환 시킬 수 있습니다.   

* * *

## How to Use   

### 0. GIt clone 
(Add git directory / 디렉토리에 이 패키지를 추가)   
 ```  !git clone https://github.com/9ruddls3/CRNN_Pytorch.git ```

### 1. DataPreparing 
()   

 ``` !python /CRNN_Pytorch/prepare.py --In_dir '/content/drive/My Drive/Datasets/HS_Crop/Hearth' --Out_dir '/content/drive/My Drive/' ```


### 2. Load DataLoader, train, save trained Pytorch model(.pth)   
()   
 ```!python /content/CRNN_Pytorch/train.py --Dataloader '/content/drive/My Drive/Datasets/dataloader.pth' --Model_path '/content/drive/My Drive' ```



2020-05-20 02:29:33 start epoch 499/500: learning_rate = 1.888946593147861e-05 sequence_len = 28   
2020-05-20 02:29:33 iter 1 loss = 0.000143   
2020-05-20 02:29:40 iter 25 loss = 0.000093   
2020-05-20 02:29:48 iter 50 loss = 0.000135   
2020-05-20 02:29:55 iter 75 loss = 0.000110   
2020-05-20 02:30:03 iter 100 loss = 0.000355   
2020-05-20 02:30:10 iter 125 loss = 0.000351   
2020-05-20 02:30:10 epoch 499/500 average_loss = 0.000232

2020-05-20 02:30:10 start epoch 500/500: learning_rate = 1.2089258196146311e-05 sequence_len = 28   
2020-05-20 02:30:11 iter 1 loss = 0.000143   
2020-05-20 02:30:18 iter 25 loss = 0.000092   
2020-05-20 02:30:25 iter 50 loss = 0.000134   
2020-05-20 02:30:33 iter 75 loss = 0.000109   
2020-05-20 02:30:40 iter 100 loss = 0.000371   
2020-05-20 02:30:48 iter 125 loss = 0.000345   
2020-05-20 02:30:48 epoch 500/500 average_loss = 0.000230

