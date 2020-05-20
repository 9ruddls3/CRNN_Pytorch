# CRNN_Pytorch
This package is for my custom Dataset HearthStone img) and based on Google Colaborative.   
If you use this for your custom dataset, you can convert your image folder to torchvision.Dataloader by using prepare.py


이 CRNN 패키지는 제 데이터셋을 기반으로 진행하였고, 구글 코랩을 기반으로 진행하였습니다.   
전반적인 Hyper-Parameter는 prams.py 에서 선언되었으며, 이를 수정함으로써 튜닝하는것도 방법입니다.   
문자열 이미지데이터가 있고, 파일명을 label.확장자로 되어있다면, prepare.py 를 통해 Datalodar 로 변환 시킬 수 있습니다.   

* * *

## What is CRNN?   
Convolution layer를 이용해 이미지의 속성을 추출한 후, 추출한 이미지를 시계열 데이터로 간주하여 예측값을 predict 하는 기법   
더 명확한 개념 설명은 아래 링크에서 확인 가능    

In a very simple way, it extracts the properties of an image using a convolution layer, and then predicts the predicted value by considering the extracted image as time series data.
You can see a clearer conceptual explanation in the link below, but that article is written in Korean. :(   

https://medium.com/qandastudy/crnn-%EB%85%BC%EB%AC%B8-%EC%86%8C%EA%B0%9C-%EB%B0%8F-%EA%B5%AC%ED%98%84-60a8cbf9bbe5   


## How to Use   

### Requirement   
1. python == 3.6.9   
2. torch == 1.5.0+cu101
3. torchvision == 0.6.0+cu101   
4. PIL == 7.0.0   
5. tqdm == 4.41.1    
   
### 0. GIt clone   

 ```  !git clone https://github.com/9ruddls3/CRNN_Pytorch.git ```  

디렉토리에 이 패키지를 추가   

Add git directory    

### 1. DataPreparing 

 ``` !python /CRNN_Pytorch/prepare.py --In_dir (path Image Folder) --Out_dir (path you want save Dataloader(.pth)) ```   

인식할 문자열 이미지 데이터셋이 포함된 폴더를 로드하여, 데이터 로더로 변환   
변환할 때, 이미지의 규격을 (224,224)로 Resize하며, 정규화과정을 transform 과정에 포함   

Load the string image data to be recognized and convert it into a Dataloader   
When transforming, the size of the image was resized to (224,224), and the normalization process was included in the transform process.   

In_dir는 이미지 파일들이 label1.jpg, label2.png, label3.jpg 으로 저당되어있는 폴더의 디렉토리 위치로 하며   
(이미지 외 다른파일 변환 시 transoform 과정에서 에러), Out_dir는 Torchvision.Dataloader를 .pth 바이너리 파일로 저장   

In_dir is the directory location of the folder where image files are saved as label1.jpg, label2.png, label3.jpg.   
(Error in transoform process when converting files other than images), Out_dir saves Torchvision.Dataloader as a .pth binary file.   

저장되는 Dataloader.dataset의 갯수 : (폴더 내 이미지 파일 갯수 // batch size(16)) x batch size   
(batch size로 나눈 나머지가 있으면, 학습시 input error가 발생 할 수 있기 때문)   

The number of saved Dataloader.dataset is (number of image files in folder // batch size (16)) x batch size.   
(If there is a remainder divided by batch size, input error may occur during learning)

### 2. Load DataLoader, train, save trained Pytorch model(.pth)   
   
 ```!python /content/CRNN_Pytorch/train.py --Dataloader (Dataloader Path) --Model_path (path you want save trained model(.pth)) ```   

Dataloader를 로드한 후, 학습을 진행하는 스크립트, 전반적인 Hyper-parameter는 pamrm.py에서 확인 가능   
epoch 수는 500회로 진행하였고, itteration 25회마다 ctcloss를 출력  
epoch당 소요시간을 확인하여, 학습 종료시간 예측 가능   

This is a script that loads the Dataloader and learns it. The overall Hyper-parameter can be found in pamrm.py.   
The number of epochs was 500, and the ctcloss rate was shown every 25 itterations.   
You can predict the end time of learning by checking the time required per epoch.   


## My Expriments Result  
이 패키지를 이용해서 하스스톤 카드 이미지 데이터 2000개로 학습하였으며,   
해상도 통일, 이름이 표시된 영역을 자르는 사전 작업을 거치고 나서 해당 패키지를 기반으로 진행   

Using this package, I learned with 2000 Hearthstone card image data.
After unification of resolution and pre-cutting of the area marked with names,  proceeded based on the package.

google colab 기준으로, gpu 처리까지 포함하여 약 5시간 20분의 학습시간이 걸렸으며 마지막 Epoch에서의 학습 성능은 아래같음   

Based on google colab, it took about 5 hours and 20 minutes of learning time including gpu processing, and the learning performance in the last epoch is as follows.   

Google colab H/W Spec : CPU:Intel(R) Xeon(R) CPU @ 2.30GHz, RAM:13GB, GPU:Tesla P100   
   
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


## List to do
1. 학습된 모델을 로드, 테스트, 예측   

Load trained model and test, prediction   

2. feature map을 출력하여, parameter가 인식하는 영역이 인간의 것과 유사한지 확인   

Print out the feature map, and check if the area recognized by the parameter is similar to the human's   

3. 새로운 이미지들을 다시 수집한 다음, 검증데이터셋으로 하여 학습모델 성능 검증 (나의 개인 프로젝트)   

After collecting new images again, verifying the performance of the learning model using the verification data set   
(My personal project)   


## References
An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition :    https://arxiv.org/abs/1507.05717

CRNN 논문 소개 : https://www.theteams.kr/teams/536/post/70322   

CRNN pytorch 구현 깃 : https://github.com/cuppersd/crnn-pytorch
