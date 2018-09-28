# -*-coding:utf-8 -*-
import pandas as pd

df_affinity_train=pd.read_csv('./df_affinity_train.csv')#165084
df_affinity_test=pd.read_csv("./df_affinity_test_toBePredicted.csv")#41383
df_affinity_test['Ki']=-1


Finger_feature=pd.read_csv("./feature/Finger_feature.csv")
molecule_feature=pd.read_csv("./feature/molecule_feature.csv")
protein_sequence_feature=pd.read_csv("./feature/protein_sequence.csv")
print('开始拼接')
train_feat=pd.merge(df_affinity_train,Finger_feature,on='Molecule_ID',how='left')
train_feat=pd.merge(train_feat,molecule_feature,on='Molecule_ID',how='left')
train_feat=pd.merge(train_feat,protein_sequence_feature,on='Protein_ID',how='left')
train_feat.to_csv('./train_feat.csv',index=False)
print('训练集生成')
test_feat=pd.merge(df_affinity_test,Finger_feature,on='Molecule_ID',how='left')
test_feat=pd.merge(test_feat,molecule_feature,on='Molecule_ID',how='left')
test_feat=pd.merge(test_feat,protein_sequence_feature,on='Protein_ID',how='left')
test_feat.to_csv('./test_feat.csv',index=False)
print('测试集生成')
