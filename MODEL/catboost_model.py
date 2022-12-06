import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from collections import defaultdict
import os, random


from scipy import sparse
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.init import normal_
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

import plotnine
from plotnine import *
from catboost import CatBoostRegressor



def category(df):
    df['sex']=df['profile_id'].map(id2sex)
    df['age']=df['profile_id'].map(id2age)
    df['pr1']=df['profile_id'].map(id2pr1)
    df['pr2']=df['profile_id'].map(id2pr2)
    df['pr3']=df['profile_id'].map(id2pr3)
    df['ch1']=df['profile_id'].map(id2ch1)
    df['ch2']=df['profile_id'].map(id2ch2)
    df['ch3']=df['profile_id'].map(id2ch3)
    df['genre_large']=df['album_id'].map(id2genre)
    df['sex_age']=df.apply(lambda x: str(x['sex'])+str(x['age']//3), axis=1)

def idx2album_id(x):
    return [idx2watched[idx] for idx in x]

# 필요한 데이터 불러오기
data_path = '/opt/ml/uplus/data'
saved_path = '../saved'
output_path = '../submission'

history_df = pd.read_csv(os.path.join(data_path, 'history_data.csv'), encoding='utf-8')
profile_df = pd.read_csv(os.path.join(data_path, 'profile_data.csv'), encoding='utf-8')
meta_df = pd.read_csv(os.path.join(data_path, 'meta_data.csv'), encoding='utf-8')
meta_p_df = pd.read_csv(os.path.join(data_path, 'meta_data_plus.csv'), encoding='utf-8')
end_df = pd.read_csv(os.path.join(data_path, 'watch_e_data.csv'), encoding='utf-8')

# album과 user의 정보를 추가하기 위한 dictionary 초기화
user2idx={id:i for i,id in enumerate(history_df['profile_id'].unique())}
album2idx={id:i for i,id in enumerate(history_df['album_id'].unique())}
idx2user={i:id for i,id in enumerate(history_df['profile_id'].unique())}
idx2album={i:id for i,id in enumerate(history_df['album_id'].unique())}
watched=history_df['album_id'].unique()
users=history_df['profile_id'].unique()
idx2watched={i:id for i,id in enumerate(watched)}

meta_df['cast_1']=meta_df['cast_1'].fillna('')
id2title,id2subtitle,id2genre,id2runtime,id2cast={},{},{},{},{}
for idx,row in meta_df.iterrows():
    id=row['album_id']
    id2title[id]=row['title']
    id2genre[id]=row['genre_large']
    id2runtime[id]=row['run_time']
    id2cast[id]=row['cast_1']
    id2subtitle[id]=row['sub_title']
    
profile_df = profile_df.fillna('')    
id2sex,id2age,id2pr1,id2pr2,id2pr3,id2ch1,id2ch2,id2ch3={},{},{},{},{},{},{},{}
for idx,row in profile_df.iterrows():
    id=row['profile_id']
    id2sex[id]=row['sex']
    id2age[id]=row['age']
    id2pr1[id]=row['pr_interest_keyword_cd_1']
    id2pr2[id]=row['pr_interest_keyword_cd_2']
    id2pr3[id]=row['pr_interest_keyword_cd_3']
    id2ch1[id]=row['ch_interest_keyword_cd_1']
    id2ch2[id]=row['ch_interest_keyword_cd_2']    
    id2ch3[id]=row['ch_interest_keyword_cd_3']

end_df['total_time']=end_df['album_id'].map(id2runtime)
end_df['watch_ratio']=end_df.apply(lambda x:x['watch_time']/x['total_time'])

# 4월 15일을 기준으로 하여 train/valid를 설정
train = pd.DataFrame()
train['profile_id']=end_df['profile_id']
train['album_id']=end_df['album_id']
train['watch_ratio']=end_df['watch_ratio']
train['train']=end_df['ss_id'].apply(lambda x:0 if int(str(x)[5:8])<415 else 1)

valid=train[train['train']==1].drop(['train'], axis=1)
train=train[train['train']==0].drop(['train'], axis=1)

# 시청 시간 비율이 5% 미만인 경우는 제거
# 각 user 별 각 album의 시청 비율의 평균을 학습하도록 설정
train=train[train['watch_ratio']>0.05].groupby(['profile_id','album_id'], as_index=False).apply(lambda x: x['watch_ratio'].mean())
train.columns=['profile_id','album_id','watch_ratio']
valid=valid[valid['watch_ratio']>0.05].groupby(['profile_id','album_id'], as_index=False).apply(lambda x: x['watch_ratio'].mean())
valid.columns=['profile_id','album_id','watch_ratio']




# negativa sampling
cnt=[]
log=train.groupby('profile_id')['album_id'].unique()
tmp=pd.DataFrame({'profile_id':[],'sub_title':[], 'run_time':[], 'cast':[],'watch_ratio':[],'album_id':[]})
pro=[]
alb=[]
negative=5
for i in users:
    try:
        negative_len=negative*len(log[i])
        tmp_set=list(set(watched)-set(log[i]))
        neg_item_ids = np.random.choice(len(tmp_set), min(negative_len, len(tmp_set)), False)
        tmp_profile=[i for _ in range(len(neg_item_ids))]
        tmp_album=[tmp_set[j] for j in neg_item_ids]
        pro+=tmp_profile
        alb+=tmp_album
    except:
        cnt.append(i)

tmp['profile_id']=pro
tmp['album_id']=alb
tmp['sub_title']=tmp['album_id'].map(id2subtitle)
tmp['run_time']=tmp['album_id'].map(id2runtime)
tmp['cast']=tmp['album_id'].map(id2cast)
tmp['watch_ratio']=0

train['sub_title']=train['album_id'].map(id2subtitle)
train['run_time']=train['album_id'].map(id2runtime)
train['cast']=train['album_id'].map(id2cast)
valid['sub_title']=valid['album_id'].map(id2subtitle)
valid['run_time']=valid['album_id'].map(id2runtime)
valid['cast']=valid['album_id'].map(id2cast)

train_=pd.concat([train,tmp])

train_

cat_model = CatBoostRegressor(
            loss_function = 'RMSE',
            iterations = 100000,
            learning_rate=0.001,
            eval_metric = 'RMSE',
            task_type='GPU',
            verbose=1000,
            max_depth=6
        )

test=pd.DataFrame()
test['profile_id']=[3 for _ in watched]
test['album_id']=watched
test['sub_title']=test['album_id'].map(id2subtitle)
test['run_time']=test['album_id'].map(id2runtime)
test['cast']=test['album_id'].map(id2cast)

#train_,valid,test에 user, album 정보 추가
category(test)
category(train_)
category(valid)

#학습 및 예측
cat_model.fit(
            train_.drop(['watch_ratio','profile_id','album_id','sex','age'], axis=1), train_['watch_ratio'], 
            cat_features = [0,2,3,4,5,6,7,8,9,10],
            eval_set = [(valid.drop(['watch_ratio','profile_id','album_id','sex','age'], axis=1), valid['watch_ratio'])],
            early_stopping_rounds=200
        )


test=test.drop(['album_id'],axis=1)
predicted_list=[]
cnt=0
for i in users:
    test['profile_id']=[i for _ in range(len(watched))]
    test['sex']=test['profile_id'].map(id2sex)
    test['age']=test['profile_id'].map(id2age)
    test['sex_age']=test.apply(lambda x: str(x['sex'])+str(x['age']//3), axis=1)
    test['pr1']=test['profile_id'].map(id2pr1)
    test['pr2']=test['profile_id'].map(id2pr2)
    test['pr3']=test['profile_id'].map(id2pr3)
    test['ch1']=test['profile_id'].map(id2ch1)
    test['ch2']=test['profile_id'].map(id2ch2)
    test['ch3']=test['profile_id'].map(id2ch3)
    test['genre_large']=test['profile_id'].map(id2genre)
    test=test.drop(['profile_id','sex','age'], axis=1)
    ret=cat_model.predict(test)
    predicted_list.append(ret.argsort()[:-25:-1])
    if cnt%1000==0:
        print('predict '+str(cnt)+'/8311 complete')
    cnt+=1

#각 user별 top-25 상품을 저장
top_catboost=pd.DataFrame()
top_catboost['profile_id']=users
top_catboost['predicted_list']=predicted_list
top_catboost['predicted_list']=top_catboost['predicted_list'].apply(idx2album_id)

top_catboost.to_csv(os.path.join(output_path, 'catboost.csv'), index = False)