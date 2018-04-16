# -*-coding:utf-8 -*-
import pandas as pd
import sys
train_feat=pd.read_csv('./train_feat.csv')
test_feat=pd.read_csv('./test_feat.csv')

#将缺失值nan补为0
train_feat=train_feat.fillna(0)
test_feat=test_feat.fillna(0)

label1=pd.DataFrame()
label2=pd.DataFrame()
label1['Ki']=train_feat['Ki']
label1.to_csv('./latest_data/label1.csv',index=False)
label2['Ki']=test_feat['Ki']
label2.to_csv('./latest_data/label2.csv',index=False)

train_feat=train_feat.drop('Ki',axis=1)
train_feat=train_feat.drop('Protein_ID',axis=1)
train_feat=train_feat.drop('Molecule_ID',axis=1)
train_feat.to_csv('./latest_data/train_data.csv')

test_feat=test_feat.drop('Ki',axis=1)
test_feat=test_feat.drop('Protein_ID',axis=1)
test_feat=test_feat.drop('Molecule_ID',axis=1)
test_feat.to_csv('./latest_data/test_data.csv')

print("删除没用的信息")
print(train_feat.columns.size)  #列数315
print(train_feat.iloc[:,0].size) #行数165084
print(test_feat.columns.size)  #列数315
print(test_feat.iloc[:,0].size) #行数41383



