import pandas as pd

dataset=pd.read_csv('mealplan2.csv').values

data=dataset[:,0:4]
target=dataset[:,4]

from sklearn.model_selection import train_test_split
#dataset splitting function

train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.1)

#from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

algorithm=SVC(kernel='linear')
#loading the SVM algorithm into "algorithm"

algorithm.fit(train_data,train_target)
#training

result=algorithm.predict(test_data)
#testing

print('Actual Target:',test_target)
print('Predicted Target:',result)

from sklearn.metrics import accuracy_score

acc=accuracy_score(test_target,result)

print('Accuracy:',acc)

import joblib

joblib.dump(algorithm,'Meal_Plan_SVM_model.sav')