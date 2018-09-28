# -*-coding:utf-8 -*-
import numpy as np
import pandas as pd
import re
from gensim.models import Word2Vec

df_protein_train=pd.read_csv('./df_protein_train.csv')#1653
#df_protein_train['seq_len']=df_protein_train['Sequence'].apply(len)
df_protein_test=pd.read_csv('./df_protein_test.csv')#414
#df_protein_test['seq_len']=df_protein_test['Sequence'].apply(len)

df_molecule=pd.read_csv('./df_molecule.csv')#111216
# df_molecule['fingerprint_len']=df_molecule['Fingerprint'].apply(len) #167

df_affinity_train=pd.read_csv('./df_affinity_train.csv')#165084
df_affinity_test=pd.read_csv("./df_affinity_test_toBePredicted.csv")#41383
df_affinity_test['Ki']=-1

#将抽象数据整合在一起以便处理
protein_concat=pd.concat([df_protein_train,df_protein_test])
all_data_concat=pd.concat([df_affinity_train,df_affinity_test])

print ('读取蛋白质ID和信息完毕')
#df_molecule_avg=df_affinity_train.groupby('Molecule_ID',as_index=False)['Ki'].agg({'Ki_avg':'mean'})
#以分子ID相同的为一类，取平均值

# print('开始拼接')
#
# df1_train=pd.merge(df_affinity_train,df_protein_train,on=['Protein_ID'],how='left')
# df1_test=pd.merge(df_affinity_test,df_protein_test,on=['Protein_ID'],how='left')
print("蛋白质序列特征") #将蛋白质序列每三个一组进行n维向量化
def protein_sequence_feature(df):
    n=128
    sequence=list(df['Sequence'])
    texts=[[word for word in re.findall(r'.{3}',document)] for document in sequence]
    model=Word2Vec(texts,size=n,window=4,min_count=1,negative=3,sg=1,hs=1,workers=4)
    vectors=pd.DataFrame([model[word] for word in (model.wv.vocab)])
    vectors['word']=list(model.wv.vocab)
    vectors.columns=['vec_{0}'.format(i) for i in range(n)]+['word']
    wide_vec=pd.DataFrame()
    result1=[]
    aa=list(protein_concat['Protein_ID'])
    for i in range(len(texts)):
        result2=[]
        for w in range(len(texts[i])):
            result2.append(aa[i])
        result1.extend(result2)
    wide_vec['ID']=result1

    result1=[]
    for i in range(len(texts)):
        result2=[]
        for w in range(len(texts[i])):
            result2.append(texts[i][w])
        result1.extend(result2)
    wide_vec['word']=result1

    del result1
    wide_vec=wide_vec.merge(vectors,on='word',how='left')
    wide_vec=wide_vec.drop('word',axis=1)
    wide_vec.columns=['Protein_ID']+["vec_{0}".format(i) for i in range(n)]
    del vectors

    name=["vec_{0}".format(i) for i in range(n)]
    sequence_feat=pd.DataFrame(wide_vec.groupby(['Protein_ID'])[name].agg('mean')).reset_index()
    sequence_feat.columns=['Protein_ID']+['mean+ci_{0}'.format(i) for i in range(n)]
    return sequence_feat


print("获取指纹特征")
def Finger_feature(df):
    f1=[]
    for i in range(len(df)):
        f1.append(df['Fingerprint'][i].split(','))
    Finger_feat=pd.DataFrame(f1)
    #Finger_feat=Finger_feat.astype('int')
    Finger_feat.columns=["Fingerprint_{0}".format(i) for i in range(167)]
    Finger_feat['Molecule_ID']=df_molecule['Molecule_ID']
    return Finger_feat

print("获取molecule中除了指纹以外其他的特征")
def molecule_feature(df):
    molecule_feat=df.drop('Fingerprint',axis=1)
    return molecule_feat

print('拼接全部数据和指纹信息以及molecule的其他信息')
Finger_feature=Finger_feature(df_molecule)
molecule_feature=molecule_feature(df_molecule)
molecule_feature.to_csv('./feature/molecule_feature.csv',index=False)
Finger_feature.to_csv('./feature/Finger_feature.csv',index=False)

print('拼接序列向量化特征')
protein_sequence_feature=protein_sequence_feature(protein_concat)
protein_sequence_feature.to_csv('./feature/protein_sequence.csv',index=False)








