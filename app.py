from flask import Flask, render_template, redirect, url_for, request ,session,jsonify
import mysql.connector
import joblib
import pytesseract
from ruwanthi import proccessImg

app = Flask(__name__)


#mysql connection
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="",
  database="hart"
)

mycursor = mydb.cursor()


#kalpana

import pandas as pd
df=pd.read_csv('cardio_dataset.csv').values
data=df[:,0:8]
target=df[:,8]

from sklearn.preprocessing import MinMaxScaler
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



@app.route("/")
def main():
    return render_template('app.html')

@app.route('/logAdmin') 
def logAdmin():    

    return render_template('LoginAdmin.html')

@app.route('/logDoc') 
def logDoc():    

    return render_template('LoginDoctor.html')

@app.route('/logPat') 
def logPat():    

    return render_template('LoginPatient.html')

@app.route('/regPat') 
def regPat():    

    return render_template('RegisterPatient.html')

@app.route('/regDoc') 
def regDoc():    

    return render_template('RegisterDoctor.html')

@app.route('/about') 
def about():    

    return render_template('about.html')

@app.route('/contact') 
def contact():    

    return render_template('contact.html')

@app.route("/adminHome")
def adminHome():
    
    return render_template('ViewDoctors.html')

@app.route('/disease') 
def disease():    
    
    return render_template('disease.html')

@app.route('/manPatient') 
def manPatient():    
    
    return render_template('ViewPatient.html')


@app.route("/doctorHome")
def doctorHome():
    
    return render_template('ViewPatient.html')


@app.route('/viewPatients') 
def viewPatients():    

    return render_template('ViewPatient.html')

@app.route('/viewReports') 
def viewReports():    

    return render_template('ViewPatient.html')

@app.route("/patientHome")
def patientHome():
    
    return render_template('Dtree.html')

@app.route('/diseasePatient') 
def diseasePatient():    

    return render_template('disease.html')

@app.route('/exercises') 
def exercises():    

    return render_template('exercises.html')

@app.route('/risk') 
def risk():    

    return render_template('risk.html')


@app.route('/chat') 
def chat():    

    return render_template('chatbot.html')

@app.route('/meal') 
def meal():    

    return render_template('meal_plan.html',result ="enter data")

@app.route('/logout') 
def logout():    

    return render_template('app.html')


@app.route('/loginAdmin',methods = ['POST', 'GET'])
def loginAdmin():
    
   if request.method == 'POST':
      username = request.form['uname']
      pw = request.form['pwd']
              
   cursor = mydb.cursor(buffered=True)
   sql_select_query = "select * from admin where adminId = %s and Password = %s"
   cursor.execute(sql_select_query, (username,pw))
   record = cursor.fetchall()
   if record is None:
       return redirect(url_for('logAdmin'))
   else:
       for x in record:
           if x is None:
               return redirect(url_for('logAdmin')) 
           else:    
               #session['Uid']=record[0] 
               return redirect(url_for('adminHome'))  
   
    
@app.route('/loginPatient',methods = ['POST', 'GET'])
def loginPatient():

   if request.method == 'POST':
      username = request.form['uname']
      pw = request.form['pwd']
              
   cursor = mydb.cursor(buffered=True)
   sql_select_query = "select * from patient where email = %s and Password = %s"
   cursor.execute(sql_select_query, (username,pw))
   record = cursor.fetchall()
   if record is None:
       return redirect(url_for('logPat'))
   else:
       for x in record:
           if x is None:
               return redirect(url_for('logPat')) 
           else:    
               #session['Uid']=record[0] 
               return redirect(url_for('patientHome'))

 
@app.route('/loginDoctor',methods = ['POST', 'GET'])
def loginDoctor():

   if request.method == 'POST':
      username = request.form['uname']
      pw = request.form['pwd']
              
   cursor = mydb.cursor(buffered=True)
   sql_select_query = "select * from doctor where drn = %s and Password = %s"
   cursor.execute(sql_select_query, (username,pw))
   record = cursor.fetchall()
   if record is None:
       return redirect(url_for('logDoc'))
   else:
       for x in record:
           if x is None:
               return redirect(url_for('logDoc')) 
           else:    
               #session['Uid']=record[0] 
               return redirect(url_for('doctorHome'))  

@app.route('/registerDoc',methods = ['POST', 'GET'])
def registerDoc():
    
   if request.method == 'POST':
      drn = request.form['drn']
      password = request.form['password']
      name = request.form['name']
      spec = request.form['spec']
      gender = request.form['gender']
      hospital = request.form['hospital']
      tel_no = request.form['tel_no']
              
   cursor = mydb.cursor(buffered=True)
   sql = """INSERT INTO doctor (drn, password,name,spec,gender,hospital,tel_no) VALUES (%s, %s ,%s,%s,%s,%s,%s)"""
   val = (drn,password,name,spec,gender,hospital,tel_no)
   cursor.execute(sql, val)
   mydb.commit() 
    
   return render_template('LoginDoctor.html')

@app.route('/registerPat',methods = ['POST', 'GET'])
def registerPat():
    
   if request.method == 'POST':
      email = request.form['email']
      password = request.form['password']
      name = request.form['name']
      gender = request.form['gender']
      dob = request.form['dob']
      tel_no = request.form['tel_no']
              
   cursor = mydb.cursor(buffered=True)
   sql = """INSERT INTO patient (email,password,name,gender,dob,tel_no) VALUES (%s, %s ,%s,%s,%s,%s)"""
   val = (email,password,name,gender,dob,tel_no)
   cursor.execute(sql, val)
   mydb.commit() 
    
   return render_template('LoginPatient.html')


@app.route('/diseaseType') 
def diseaseType():    

    return render_template('Disease_type.html')


@app.route('/predict_dis',methods=['GET','POST']) 
def predict_dis():    

    data=request.form
    val1=eval(data['v1'])   #converting text into float
    val2=eval(data['v2'])
    val3=eval(data['v3'])
    val4=eval(data['v4'])
    val5=eval(data['v5'])
    val6=eval(data['v6'])
    val7=eval(data['v7'])
    val8=eval(data['v8'])
    val9=eval(data['v9'])
    val10=eval(data['v10'])
    val11=eval(data['v11'])
    val12=eval(data['v12'])
    val13=eval(data['v13'])
    val14=eval(data['v14'])
    val15=eval(data['v15'])
    val16=eval(data['v16'])
    val17=eval(data['v17'])
    val18=eval(data['v18'])
    val19=eval(data['v19'])
    val20=eval(data['v20'])
    val21=eval(data['v21'])
    val22=eval(data['v22'])
    val23=eval(data['v23'])
    val24=eval(data['v24'])
    val25=eval(data['v25'])
    val26=eval(data['v26'])
    val27=eval(data['v27'])
    val28=eval(data['v28'])
    val29=eval(data['v29'])
    val30=eval(data['v30'])
    val31=eval(data['v31'])
    val32=eval(data['v32'])
    val33=eval(data['v33'])
    val34=eval(data['v34'])
    val35=eval(data['v35'])
    val36=eval(data['v36'])
    

    algorithm=joblib.load('Type_of_the_Disease.sav')
    #loading the trained algorithm
    result=algorithm.predict([[val1,val2,val3,val4,val5,val6,val7,val8,val9,val10,val11,val12,val13,val14,val15,val16,val17,val18,val19,val20,val21,val22,val23,val24,val25,val26,val27,val28,val29,val30,val31,val32,val33,val34,val35,val36]])

    #print(val1,val2,val3,val4,result)

    return "Type of The Heart Disease:"+str(result[0])


@app.route('/kalpana',methods = ['POST', 'GET'])
def kalpana():
    
    model=load_model()

    User_json = request.json
    
    if request.method == 'POST':
      age = int(request.form['age'])
      gender = int(request.form['gender'])
      med = int(request.form['blood'])
      dia = int(request.form['diabet'])
      smoke = int(request.form['smoke'])
      sbp = int(request.form['sbp'])
      
    tc,hdl=proccessImg  

    test_data=[gender,age,tc,hdl,sbp,smoke,med,dia]

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


@app.route('/predict_mealPlan',methods=['GET','POST']) 
def predict_mealPlan():    

    data=request.form
    val1=eval(data['gender'])   #converting text into float
    val2=eval(data['age'])
    val3=eval(data['bmi'])
    val4=eval(data['risk'])

    algorithm=joblib.load('Meal_Plan_SVM_model.sav')
    #loading the trained algorithm
    result=algorithm.predict([[val1,val2,val3,val4]])

    #print(val1,val2,val3,val4,result)

    #return "PREDICTED MEAL PLAN:"+str(int(result[0]))
    return render_template("meal_plan.html",result = str(int(result[0])))    
        
if __name__ == "__main__":
    app.run()


