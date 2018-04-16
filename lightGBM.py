# -*-coding:utf-8 -*-
import numpy as np
import pandas as pd
import lightgbm as lgb
import sys



train_data=pd.read_csv('./latest_data/train_data.csv')
train_data=train_data.astype('double')
test_data=pd.read_csv('./latest_data/test_data.csv')
test_data=test_data.astype('double')
label1=pd.read_csv('./latest_data/label1.csv')

label2=pd.read_csv('./latest_data/label2.csv')

test_tobepredicted=pd.read_csv('./df_affinity_test_toBePredicted.csv')


# train=lgb.Dataset(train_data,label=label1)
# test=lgb.Dataset(test_data,label=label2,reference=train)
lgb_train=lgb.Dataset(train_data,label1)
print('使用LIGHTBGM训练')
params={
    'boosting_type':'gbdt',#训练方式
    'objective':'regression_l2',#目标
    'metric':'l2',#损失函数
    'min_child_weight':3,#叶子上的最小样本数
    'num_leaves':2**5,
    'lambda_l2':10,
    'subsample':0.7,
    'colsample_bytree': 0.7,
    'colsample_bylevel':0.7,
    'learning_rate':0.05,
    'tree_method': 'exact',
    'seed':2017,
    'nthread':12,
    'silent': True
    }

num_round=3000
gbm=lgb.train(params,lgb_train,num_round,verbose_eval=50)

print('开始预测')
preds_sub=gbm.predict(test_data)

test_tobepredicted['Ki']=preds_sub
test_tobepredicted.to_csv('result1.csv',index=False)

# with open("result1.csv", "w") as f:
#     sys.stdout = f
#     print "Protein_ID,Molecule_ID"
#     for index, Protein_ID in enumerate(user_ids):
#         print "{},{}".format(userid, y[index])