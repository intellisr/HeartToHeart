from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

import pandas as pd
df=pd.read_csv('cardio_dataset.csv').values
data=df[:,0:8]
target=df[:,8]

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

target=np.reshape(target, (-1,1))
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_x.fit(data)
scaler_y.fit(target)
xscale=scaler_x.transform(data)
yscale=scaler_y.transform(target)

from keras import backend as K
import keras.models as models
import keras.layers as layers
import keras.optimizers as optimizers
from keras.layers import Dropout

def load_model():

    model = models.Sequential()
    model.add(layers.Dense(128, input_dim=8, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(layers.Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(optimizer='adam',loss='mse',metrics=['mse','mae'])

    model.load_weights('Predictions.h5')
    return model

@app.route('/')
def student():
   return render_template('student.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():

    model=load_model()

    User_json = request.json

    sex = int(User_json['sex'])
    age = int(User_json['age'])
    tc = int(User_json['tc'])
    hdl = int(User_json['hdl'])
    sbp = int(User_json['sbp'])
    smoke = int(User_json['smoke'])
    med = int(User_json['med'])
    dia = int(User_json['dia'])
    
    # if request.method == 'POST':
    #     result = request.json
    #     print(result)
    #     sex=int(result["sex"])
    #     age=int(result["age"])
    #     tc=int(result["tc"])
    #     hdl=int(result["hdl"])
    #     sbp=int(result["sbp"])
    #     smoke=int(result["smoke"])
    #     med=int(result["med"])
    #     dia=int(result["dia"])

    test_data=[sex,age,tc,hdl,sbp,smoke,med,dia]

    print(test_data)
    
    test_data=scaler_x.transform([test_data])
    ped_result=model.predict(test_data)
    #ped_result={'Risk Level':ped_result[0][0]}
    K.clear_session()
    reults = [
    {
        "ped_result":float(ped_result)
    }
    ]
    return jsonify(results=reults)

        #return render_template("result.html",result = ped_result)

if __name__ == "__main__":
    app.run()