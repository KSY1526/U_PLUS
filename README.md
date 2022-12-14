## 팀원

| <img src="https://user-images.githubusercontent.com/64895794/200263288-1d77b5f8-ed79-4548-9bc1-01aec2474aaa.png" width=200> | <img src="https://user-images.githubusercontent.com/64895794/200263509-9f564042-6da7-4410-a820-c8198037b0b3.png" width=200> | <img src="https://user-images.githubusercontent.com/64895794/200263509-9f564042-6da7-4410-a820-c8198037b0b3.png" width=200> |
| 
| :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: |
|                                           [김성연](https://github.com/KSY1526)                                            |                                           [구진범](https://github.com/jb5341)                                            |                                            [이환주](https://github.com/Leehj0530)                                            |


## 1. 간단한 프로젝트 소개



## 2. 하드웨어 스펙/운영체제 및 플랫폼 사용

Upstage V100

## 3. 소스코드 구조 설명

EDA/EDA.ipynb : 전반적인 EDA 내용을 담고 있습니다.  
EDA/personal/ : 개인적으로 진행된 실험적인 EDA 내용 입니다.  
input : 사용하는 데이터가 저장되어 있는 파일입니다.  
MODEL/catboost_model.py : catboost 추천 모델을 돌리는 파일입니다.  
MODEL/LightGCN_EMB_MODEL.ipynb : LightGCN 추천 모델을 돌리는 파일입니다.  
MODEL/MF_Model.ipynb : MF 추천 모델을 돌리는 파일입니다.  
MODEL/RuleBase_Age.ipynb : RuleBase + Age그룹 별 추천 모델을 돌리는 파일입니다. 이 모델이 우리 팀의 최종 제출 모델입니다. 
saved/ : 모델이 중간 저장되는 장소입니다.  
submission/ : 모델 최종 산출물이 저장되는 장소입니다.  
requirement.txt : catboost, mf, rulebase 모델 가상환경 구축을 위한 파일입니다.  
install.sh : lightgcn 모델 가상환경 구축을 위한 파일입니다.  

## 4. 결과 재현 방법

### LightGCN
conda create -n gcn python=3.10  
conda activate gcn  
chmod +x install.sh  
./install.sh  
python -m ipykernel install --user --name gcn --display-name gcn  
이후 gcn 커널로 MODEL/LightGCN_EMB_MODEL.ipynb 실행  

### catboost, mf, rulebase
conda create -n recsys python=3.9  
conda activate recsys  
pip install -r requirements.txt  
catboost : python MODEL/catboost_model.py 입력  
mf : recsys 커널로 MODEL/MF_Model.ipynb 실행  
rulebase : recsys 커널로 MODEL/RuleBase_Age.ipynb 실행  
