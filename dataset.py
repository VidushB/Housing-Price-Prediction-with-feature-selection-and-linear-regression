import os
import pandas as pd
root = '/content/drive/My Drive/'
os.chdir(root)
f1=root+ '/train.csv'
f2=root+ '/test.csv'
df_train=pd.read_csv(f1)
df_test=pd.read_csv(f2)
ID_count=df_train.shape[0] # The number of IDs
for column in df_train.columns:
  #print(column, "    ", df_train[column].isnull().sum())  #259 nulls in lot frontage is high. But its an important feature. #690 for fire place quality is too high. So drop all where more than 45% are NaN
  if df_train[column].isnull().sum()>=0.47*ID_count:
    df_train.drop(column,axis=1, inplace=True)
    df_test.drop(column,axis=1,inplace=True)
for column in df_train.columns:
  if df_train[column].value_counts().values[0]>.85*ID_count:
    df_train.drop(column,axis=1, inplace=True)
    df_test.drop(column,axis=1,inplace=True)
df_train.drop(["Id"],axis=1,inplace=True)
df_test.drop(["Id"],axis=1,inplace=True)
categorical=["MSSubClass","MSZoning","Neighborhood", "LotShape","LotConfig","BldgType","HouseStyle","OverallQual","OverallCond","RoofStyle","Exterior1st","Exterior2nd","MasVnrType","ExterQual","Foundation","BsmtQual","BsmtExposure","BsmtFinType1","HeatingQC","BsmtFullBath","FullBath","HalfBath","BedroomAbvGr","KitchenQual","TotRmsAbvGrd","Fireplaces","GarageType","GarageFinish","GarageCars","MoSold","SaleCondition"]
continuous=["LotFrontage","LotArea","YearBuilt","YearRemodAdd","MasVnrArea","BsmtFinSF1","BsmtUnfSF","TotalBsmtSF","1stFlrSF","2ndFlrSF","GrLivArea","GarageYrBlt","GarageArea","WoodDeckSF","OpenPorchSF","YrSold","SalePrice"]
for column in df_test.columns:
    if column in categorical:
        df_train[column] = df_train[column].fillna(value=df_train[column].value_counts().index[0])
        df_test[column] = df_test[column].fillna(value=df_test[column].value_counts().index[0])
    else:
        df_train[column] = df_train[column].fillna(value=df_train[column].mean()) 
        df_test[column] = df_test[column].fillna(value=df_test[column].mean())
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
for column in continuous:
    df_train[column] = scaler.fit_transform(df_train[column].values.reshape(-1,1))
    if column !="SalePrice":
      df_test[column] = scaler.transform(df_test[column].values.reshape(-1,1))
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown='ignore') #Creating an encoder
enc.fit(df_train[categorical]) # Calling a fit
df_cat=enc.transform(df_train[categorical])
df_cat.columns=enc.get_feature_names(categorical)

df_cat=pd.DataFrame.sparse.from_spmatrix(df_cat)
df_cat.columns=enc.get_feature_names(categorical)
df_train=df_train.drop(categorical,axis=1)
df_train=pd.concat([df_train,df_cat],axis=1)
df_train

dtest_cat = df_test[categorical]
dtest_cat = enc.transform(df_test[categorical])
dtest_cat = pd.DataFrame(dtest_cat.todense())
dtest_cat.columns = enc.get_feature_names(categorical)

df_test = df_test.drop(categorical, axis=1)
df_test = df_test.reset_index()
dtest_cat = dtest_cat.reset_index()
df_test = pd.concat([df_test, dtest_cat], axis=1)
df_test = df_test.drop('index', axis=1)
df_test
