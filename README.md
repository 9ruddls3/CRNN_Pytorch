# CRNN_Pytorch
This package is for my custom Dataset HearthStone img) and based on Google Colaborative.   
If you use this for your custom dataset, you can convert your image folder to torchvision.Dataloader by using prepare.py


이 CRNN 패키지는 제 데이터셋을 기반으로 진행하였고, 구글 코랩을 기반으로 진행하였습니다.   
전반적인 Hyper-Parameter는 prams.py 에서 선언되었으며, 이를 수정함으로써 튜닝하는것도 방법입니다.   
문자열 이미지데이터가 있고, 파일명을 label.확장자로 되어있다면, prepare.py 를 통해 Datalodar 로 변환 시킬 수 있습니다.   

* * *

## How to Use   

### Requirement   
1. python   
2. torch, pytorch   
3. PIL   
4. tqdm    
   
    
### 0. GIt clone 
Add git directory /   
디렉토리에 이 패키지를 추가합니다.   
 ```  !git clone https://github.com/9ruddls3/CRNN_Pytorch.git ```

### 1. DataPreparing 

 ``` !python /CRNN_Pytorch/prepare.py --In_dir (path Image Folder) --Out_dir (path you want save Dataloader(.pth)) ```   

인식할 문자열 이미지 데이터셋이 포함된 폴더를 로드하여, 데이터 로더로 변환합니다.   
Load the string image data to be recognized and convert it into a Dataloader   

In_dir는 이미지 파일들이 label1.jpg, label2.png, label3.jpg 으로 저당되어있는 폴더의 디렉토리 위치로 하며   
(이미지 외 다른파일 변환 시 transoform 과정에서 에러), Out_dir는 Torchvision.Dataloader를 .pth 바이너리 파일로 저장됩니다.   
In_dir is the directory location of the folder where image files are saved as label1.jpg, label2.png, label3.jpg.
(Error in transoform process when converting files other than images), Out_dir saves Torchvision.Dataloader as a .pth binary file.


저장되는 Dataloader.dataset의 갯수는 (폴더 내 이미지 파일 갯수 // batch size(16)) x batch size 가 되도록 하였습니다.   
(batch size로 나눈 나머지가 있으면, 학습시 input error가 발생 할 수 있기 때문)   
The number of saved Dataloader.dataset is (number of image files in folder // batch size (16)) x batch size.
(If there is a remainder divided by batch size, input error may occur during learning)

### 2. Load DataLoader, train, save trained Pytorch model(.pth)   
   
 ```!python /content/CRNN_Pytorch/train.py --Dataloader (Dataloader Path) --Model_path (path you want save trained model(.pth)) ```   



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

ACKNOWLEGEMEMT
An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition :    https://arxiv.org/abs/1507.05717

CRNN 논문 소개 및 구현 : https://www.theteams.kr/teams/536/post/70322   

