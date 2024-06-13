import pandas as pd
from category_encoders import TargetEncoder 
#from sklearn.preprocessing import TargetEncoder

dataset=pd.read_csv("C:/Users/DARAN/OneDrive/Codsoft/Data Science/IMDb Movies India.csv",encoding='Latin-1')

data=dataset.copy()

remove=['Name','Duration']
data=data.drop(remove,axis=1)

data.drop_duplicates(inplace=True)

cols=['Rating','Votes','Director','Actor 1','Actor 2','Actor 3']

data=data.dropna(subset=cols)

data['Year']=data['Year'].str.strip('()').astype(int)

data['Votes']=data['Votes'].str.replace(',','').astype(int)

cols=['Genre','Director','Actor 1','Actor 2','Actor 3']
target='Rating'

#data1=data.drop(data[cols],axis=1)

te=TargetEncoder(cols=cols)
encoded_columns=te.fit_transform(data[cols],data[target])

data=data.drop(data[cols],axis=1)

data=pd.concat([data,encoded_columns],axis=1)

from sklearn.model_selection import train_test_split as tts

titles=list(data.columns)
titles[0],titles[1] = titles[1],titles[0]
data=data[titles]

x=data.iloc[:, 1:7]
y=data.iloc[:, 0]

xtrain,xtest,ytrain,ytest=tts(x,y,train_size=0.8,random_state=235)

from sklearn.ensemble import GradientBoostingRegressor 
#from xgboost import XGBRegressor 
gb=GradientBoostingRegressor()

model=gb.fit(xtrain,ytrain)

prediction=gb.predict(xtest)

model_score=gb.score(xtest, ytest)

print(f"The accuracy of the model is {model_score*100:.3f}%")
