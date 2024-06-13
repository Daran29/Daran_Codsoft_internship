import pandas as pd 
import math

dataset=pd.read_csv("C:/Users/DARAN/OneDrive/Codsoft/Data Science/Titanic-Dataset.csv")

data=dataset.copy()

#DATA PREPROCESSING

cols=['PassengerId','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked']

data=data.drop(cols,axis=1)

col=['Age']

data[col]=data[col].fillna(data.mean(numeric_only=True))

from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder()
data['Sex']= le.fit_transform(data['Sex'])   

#MODEL TRAINING

x=data.iloc[:, 1:]
y=data.iloc[:, 0]

from sklearn.model_selection import train_test_split 

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=12369)

from sklearn.linear_model import LogisticRegression as lr

log_reg= lr()

model = log_reg.fit(xtrain,ytrain)

predict = log_reg.predict(xtest)
print(predict)

model_score= log_reg.score(xtest,ytest)

from sklearn.metrics import mean_squared_error as mse

rmse=math.sqrt(mse(ytest,predict))

print(f"The Root mean squared error value is {rmse:.2f}")

print(f"The Accuracy of the model is {model_score*100:.3f}%")
