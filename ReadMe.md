# Business Name Classfier by LSTM (Keras) 

한글,영어 및 특수기호가 포함된 문자열 데이터들의 카테고리를 분류하는 인공지능을 구현하는 프로젝트 입니다.   
4만여건의 학습데이터, 4천여건의 검증데이터를 기반으로 진행하였으며, 품사 및 단어 분석기는 은전한닢 프로젝트에서 배포하는 Mecab 라이브러리를 기반으로 진행하였습니다.   
학습 데이터셋에서, 각 카테고리에 포함되는 문자열들을 다 통합을 시킨 뒤 하나의 문서로써 다룹니다. 그리고 모든 문서를 명사 혹은 품사단위로 태깅하여 분류된 명사 및 품사의 기준을 공백으로 채우며, sklearn 라이브러리 중, tfidf-Vectorizer 모듈를 적용시킬 때, 토큰 단위를 공백으로 하게끔 하면서 한글문서에서 tf-idf matrix를 효과적으로 적용시킵니다.   

각 태그마다 카테고리에 관한 가중치를 담은 Matrix를 Word-Embedded-Vector로 치환합니다. 그리고 토큰:vector 로 이루어진 딕셔너리을 선언하여 Sentence Embedding을 진행합니다.   

Embedded-Sentence들을 Input data로 적용시켜 LSTM에 학습할 수 있도록 진행합니다. 적용시킨 딥러닝 모델은 LSTM이며, Adam optimizer를 적용시켜 가중치 데이터 변경 방향과, 스텝사이즈를 조절하며 진행하였습니다.   
   
   
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


### 1. Data Check and Re-structure   
   Pandas 로 데이터 확인 후, Mecab 라이브러리를 git-clone 및 설치   
            
   데이터들을 카테고리 별 문서로 통합 (ex/ 'starbucks' + '탐앤탐스' => 'starbucks 탐앤탐스')   
      
   통한된 문서들을 기준으로 Mecab 라이브러리를 통한 명사/품사 태깅   
   (ex/ 'starbucks 탐앤탐스' => ['star','bucks','탐','앤','탐스']) 
      
   생성된 리스트 기반 문서 재 통합   
   (ex/ ['star','bucks','탐','앤','탐스'] => 'star bucks 탐 앤 탐스' )
   
   
### 2. TF-Idf Vectorize(Word Embedding) & Data Preparing   
   Sklearn의 TfidfVectorizer 함수틑 통해, 각 토큰과, 문서간의 연관성을 나타내는 Matrix생성   
      
   를 얻고, 각 단어를 1x17의 vector로 치환하는


### 3. LSTM Training  





## Expriments Result  


1. 명사 단위 태깅   

|Train/Test|Cross Entropy Loss|Top-1 Accuracy(%)|Top-3 Accuracy(%)|Top-5 Accuracy(%)|F1 Score|
|------|------|---|---|---|---|
|Train|0.5275|84.89%|93.05%|96.27%|0.8484|
|Test|0.6225|82.56%|91.66%|95.49%|0.8216|


2. 품사 단위 태깅   

|Train/Test|Cross Entropy Loss|Top-1 Accuracy(%)|Top-3 Accuracy(%)|Top-5 Accuracy(%)|F1 Score|
|------|------|---|---|---|---|
|Train|0.5126|85.41%|93.43%|96.20%|0.8534|
|Test|0.6071|82.94%|91.93%|95.49%|0.8254|
