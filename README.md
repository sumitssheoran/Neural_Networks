# Neural_Networks
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df1=pd.read_csv("titanic.csv")
df2=pd.read_csv("titanictest.csv")
#Adding new column "type" to keep train and test data separate
df1["type"]="train"
df2["type"]="test"
df=pd.concat([df1,df2],ignore_index=True)
#Dropping the unnecessary columns
df.drop(["PassengerId","Ticket"],axis=1,inplace = True)
df.info()
df.isnull().sum()
names=list(df["Name"])
for i in range(len(names)):
    names[i]=names[i].split(",")[1]
    names[i]=names[i].split(".")[0]
df["Name"]=names
df["Name"].value_counts()
df["Name"].unique()

df.loc[(df["Name"]==" the Countess")|
        (df["Name"]==" Sir")|
        (df["Name"]==" Mme")|
        (df["Name"]==" Jonkheer")|
        (df["Name"]==" Don")|
        (df["Name"]==" Dona"),"Name"]=0
df.loc[df["Name"]==" Mr","Name"]=1
df.loc[df["Name"]==" Miss","Name"]=2
df.loc[df["Name"]==" Mrs","Name"]=3
df.loc[df["Name"]==" Master","Name"]=4
df.loc[df["Name"]==" Dr","Name"]=5
df.loc[df["Name"]==" Rev","Name"]=6
df.loc[(df["Name"]==" Major")|
        (df["Name"]==" Mlle")|
        (df["Name"]==" Col")|
        (df["Name"]==" Capt"),"Name"]=7
df.loc[(df["Name"]==" Lady")|
        (df["Name"]==" Ms"),"Name"]=8  
        
#Visualize no. of survivors based on their names
sns.countplot(x="Name",hue="Survived",data=df)


df["Cabin"].fillna(value="Z",inplace=True)
df["Cabin"].value_counts()
#Assemble 'Cabin' based on their initial alphabet.
for i in range(len(df["Cabin"])):
    df["Cabin"][i]=df["Cabin"][i][0]
sns.countplot(x="Cabin",hue="Survived",data=df)
df["Cabin"].value_counts().plot(kind="bar")


from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df['Cabin'] = LE.fit_transform(df['Cabin'])

df.info()
df["Embarked"].value_counts()
#Using sklearn to fill missing values.
from sklearn.impute import SimpleImputer
si=SimpleImputer(missing_values=np.nan,strategy="most_frequent")
si=si.fit(df[["Embarked"]])
df[["Embarked"]]=si.transform(df[["Embarked"]])
df['Embarked'] = LE.fit_transform(df['Embarked'])
        

si=SimpleImputer(missing_values=np.nan,strategy="median")
si=si.fit(df[["Age"]])
df[["Age"]]=si.transform(df[["Age"]])
df["Age"]=df["Age"].astype(np.float64)


df["Sex"].value_counts()
df.loc[df["Sex"]=="male","Sex"]=0
df.loc[df["Sex"]=="female","Sex"]=1



df["Fare"].fillna(value=df["Fare"].mean(),inplace=True)
df.info()
desc=df.describe() 
df.dtypes
#df.loc[(df["Age"]>=0.17) &( df["Age"]<22),"Age"]=0
#df.loc[(df["Age"]>=22) &( df["Age"]<28),"Age"]=1
#df.loc[(df["Age"]>=28) &( df["Age"]<35),"Age"]=2
#df.loc[(df["Age"]>=35) &( df["Age"]<=80),"Age"]=3

#df.loc[(df["Fare"]>=0) &( df["Fare"]<=7.8958),"Fare"]=0
#df.loc[(df["Fare"]>=7.8958) &( df["Fare"]<=14.4542),"Fare"]=1
#df.loc[(df["Fare"]>=14.4542) &( df["Fare"]<=31.275),"Fare"]=2
#df.loc[(df["Fare"]>=31.275) &( df["Fare"]<=512.329),"Fare"]=3

#Separate train and test data based on the label "type"
X_train=df.loc[:890,df.columns!="Survived"] 
X_train.drop("type",axis=1,inplace=True) 

X_test=df.loc[891:1308,df.columns!="Survived"]
X_test.drop("type",axis=1,inplace=True)    
y_train=df.loc[:890,"Survived"]        
y_test=df.loc[891:1308,"Survived"] 

from sklearn.preprocessing import StandardScaler 
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense
classifier=Sequential()
classifier.add(Dense(units=5,kernel_initializer="uniform",activation="relu",input_dim=9))
classifier.add(Dense(units=5,kernel_initializer="uniform",activation="relu"))
classifier.add(Dense(units=1,kernel_initializer="uniform",activation="sigmoid"))
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
classifier.fit(X_train,y_train,batch_size=10,epochs=1000)
y_pred=classifier.predict(X_test)
y_pred2=classifier.predict_classes(X_test)

df3=df2["PassengerId"]
df3=pd.DataFrame(df3)

df3.set_index("PassengerId",inplace=True)
df3["Survived"]=y_pred2
df3.to_csv(r'C:\Users\Admin\Desktop\ml practice\kaggle_titanic_neural_networks.csv')


