# -*-coding:utf-8 -*-
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split



train_data=pd.read_csv('./latest_data/train_data.csv')
train_data=train_data.astype('double')
test_data=pd.read_csv('./latest_data/test_data.csv')
test_data=test_data.astype('double')


label1=pd.read_csv('./latest_data/label1.csv')
label1=label1['Ki']
label2=pd.read_csv('./latest_data/label2.csv')
label2=label2['Ki']
test_tobepredicted=pd.read_csv('./df_affinity_test_toBePredicted.csv')

x_train,x_val,y_train,y_val=train_test_split(train_data,label1,test_size=0.2,random_state=100)
# train=lgb.Dataset(train_data,label=label1)
# test=lgb.Dataset(test_data,label=label2,reference=train)
xgb_train=xgb.DMatrix(x_train,label=y_train)
xgb_val=xgb.DMatrix(x_val,label=y_val)
xgb_test=xgb.DMatrix(test_data)
print('使用XGboost训练')

params={
    'n_estimators':1000,
    'max_depth':4,
    'min_child_weight':3,#叶子上的最小样本数
    'subsample':0.7,
    'colsample_bytree': 0.7,
    'colsample_bylevel':0.7,
    'learning_rate':0.05,
    'seed':2017,
    'nthread':12,
    'silent': 1
    }



#plst+=[('eval_metric','auc')]
#evallist=[(x_val,'eval'),(x_train,'train')]

num_round=3000
plst=list(params.items())
plst+= [('eval_metric', 'auc')]
evallist = [(xgb_val, 'eval'), (xgb_train, 'train')]
xgb=xgb.train(params,xgb_train,num_boost_round=num_round)
print("save model")
xgb.save_model('./model/model4.txt')

print('开始预测')
preds_sub=xgb.predict(xgb_test)

test_tobepredicted['Ki']=preds_sub
test_tobepredicted.to_csv('./result/result4.csv',index=False)

# with open("result1.csv", "w") as f:
#     sys.stdout = f
#     print "Protein_ID,Molecule_ID"
#     for index, Protein_ID in enumerate(user_ids):
#         print "{},{}".format(userid, y[index])
