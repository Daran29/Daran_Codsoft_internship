import pandas as pd

dataset=pd.read_csv('C:/Users/DARAN/OneDrive/Codsoft/Data Science/IRIS.csv')

data=dataset.copy()

cols=['sepal_length','sepal_width','petal_length','petal_width']

#from sklearn.preprocessing import TargetEncoder
from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder()

data['species']= le.fit_transform(data['species'])

x=data.iloc[:, 0:3]
y=data.iloc[:, -1]

from sklearn.model_selection import train_test_split as tts

xtrain,xtest,ytrain,ytest = tts(x,y,test_size=0.3,random_state=23)

#from sklearn.pipeline import make_pipeline
#from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#pipe = make_pipeline(StandardScaler(), LogisticRegression())
#pipe.fit(xtrain,ytrain)


lr=LogisticRegression(solver='lbfgs', max_iter=400)

model=lr.fit(xtrain, ytrain)

prediction=lr.predict(xtest)

model_score=lr.score(xtest, ytest)

from sklearn.metrics import mean_squared_error as mse
import math

rmse=math.sqrt(mse(ytest,prediction))

print(f"The accuracy of the model is {model_score*100:.3f}%")








