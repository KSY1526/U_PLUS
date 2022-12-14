## 1. 팀원 소개

| <img src="https://user-images.githubusercontent.com/79916736/207595518-c87d8c72-e1a6-4560-91c4-77487d9d34f6.png" width=200> | <img src="https://user-images.githubusercontent.com/79916736/207595901-3ea9190c-0a6f-4ee4-b609-c423a9073996.png" width=200> | <img src="https://user-images.githubusercontent.com/79916736/207594973-a16dd2e9-a332-4088-959f-673308a29e99.png" width=200> |
| :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: |
|                                           [김성연](https://github.com/KSY1526)                                            |                                           [구진범](https://github.com/jb5341)                                            |                                            [이환주](https://github.com/Leehj0530)                                            |


## 2. 간단한 프로젝트 설명
LG U+ 아이들나라 서비스 데이터를 활용하여 고객이 다음에 시청할 콘텐츠를 추천하는 대회입니다. [링크](https://stages.ai/competitions/208/overview/description)

나이 기반 Rule-base Model, NeuMF Model, LightGCN Model, CatBoost Model을 사용했습니다.

Private 기준 216팀 중 14위를 기록해 본선 진출 성공 하였습니다.

![image](https://user-images.githubusercontent.com/79916736/207597579-ea25b1de-9dcc-4cd0-bb47-177f087a66ea.png)

자세한 내용은 Presentation 폴더의 발표자료를 참고해주세요!


## 3. 하드웨어 스펙/운영체제 및 플랫폼 사용

Upstage V100

## 4. 소스코드 구조 설명

Presentation/ : 프로젝트 발표 자료가 있습니다.
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

## 5. 결과 재현 방법

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
