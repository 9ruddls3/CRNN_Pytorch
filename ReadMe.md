# Business Name Classfier by LSTM (Keras) 

한글,영어 및 특수기호가 포함된 문자열 데이터들의 카테고리를 분류하는 인공지능을 구현하는 프로젝트 입니다.   
4만여건의 학습데이터, 4천여건의 검증데이터를 기반으로 진행하였으며, 품사 및 단어 분석기는 은전한닢 프로젝트에서 배포하는 Mecab 라이브러리를 기반으로 진행하였습니다.   
학습 데이터셋에서, 각 카테고리에 포함되는 문자열들을 다 통합을 시킨 뒤 하나의 문서로써 다룹니다. 그리고 모든 문서를 명사 혹은 품사단위로 태깅하여 분류된 명사 및 품사의 기준을 공백으로 채우며, sklearn 라이브러리 중, tfidf-Vectorizer 모듈를 적용시킬 때, 토큰 단위를 공백으로 하게끔 하면서 한글문서에서 tf-idf matrix를 효과적으로 적용시킵니다.




데이터에서, 명사 및 품사 태깅을 한 

이 패키지에 쓰인 언어 및 라이브러리들은 다음과 같습니다.   

1. python3   
2. keras (tensorflow)
3. numpy   
4. pandas   
5. sklearn
6. matplotlib
7.seaborn


* * *


## Process   


### 1.Data Check and Re-structure


   

### 2. Load DataLoader, train, save trained Pytorch model(.pth)   
   


### 3. LSTM Training  





## Expriments Result  

1. 명사단위 태깅   

|Cross Entropy Loss | Top-1 Accuracy|Top-3 Accuracy|Top-5 Accuracy|F1 Score|
|------|---|---|설명|설명|설명|설명|설명|설명|설명|
|테스트1|테스트2|테스트3|설명|설명|설명|설명|설명|설명|설명|
|테스트1|테스트2|테스트3|설명|설명|설명|설명|설명|설명|설명|
|테스트1|테스트2|테스트3|설명|설명|설명|설명|설명|설명|설명|   


2. 품사 단위 태깅   

|제목|내용|설명|설명|설명|설명|설명|설명|설명|설명|
|------|---|---|설명|설명|설명|설명|설명|설명|설명|
|테스트1|테스트2|테스트3|설명|설명|설명|설명|설명|설명|설명|
|테스트1|테스트2|테스트3|설명|설명|설명|설명|설명|설명|설명|
|테스트1|테스트2|테스트3|설명|설명|설명|설명|설명|설명|설명|   







## List to do
1. 학습된 모델을 로드, 테스트, 예측   

Load trained model and test, prediction   

2. feature map을 출력하여, parameter가 인식하는 영역이 인간의 것과 유사한지 확인   

Print out the feature map, and check if the area recognized by the parameter is similar to the human's   

3. 새로운 이미지들을 다시 수집한 다음, 검증데이터셋으로 하여 학습모델 성능 검증 (나의 개인 프로젝트)   

After collecting new images again, verifying the performance of the learning model using the verification data set   
(My personal project)   


