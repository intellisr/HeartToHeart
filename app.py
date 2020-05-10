from flask import Flask, render_template, redirect, url_for, request ,session,jsonify,json
import mysql.connector
import joblib
import pytesseract
from sklearn.svm import SVC
import dialogflow
from google.api_core.exceptions import InvalidArgument
from PIL import Image
import cv2
import os
import re
import urllib.request


#mysql connection
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="",
  database="heart"
)

mycursor = mydb.cursor()

#chatbot
DIALOGFLOW_PROJECT_ID = 'heartbot-fretyc'
DIALOGFLOW_LANGUAGE_CODE = 'en-US'
GOOGLE_APPLICATION_CREDENTIALS = 'heartbot-fretyc-d6ea36d86616.json'
SESSION_ID = 'SRA123'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=GOOGLE_APPLICATION_CREDENTIALS

pytesseract.pytesseract.tesseract_cmd='C://Program Files/Tesseract-OCR/tesseract.exe'
#from ruwanthi import proccessImg

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'


#codes for chatbot
def convertTofloat(msg):
    if msg=="yes":
        return float(1)
    else:    
        return float(0)
    
def validateAnswers(intent,answer):
    if intent=="q2":
        session['q1']=convertTofloat(answer)        
    elif intent=="q3":    
        session['q2']=convertTofloat(answer)
    elif intent=="q4":    
        session['q3']=convertTofloat(answer)
    elif intent=="q5":    
        session['q4']=convertTofloat(answer)
    elif intent=="q6":    
        session['q5']=convertTofloat(answer)
    elif intent=="q7":    
        session['q6']=convertTofloat(answer)
    elif intent=="q8":    
        session['q7']=convertTofloat(answer)
    elif intent=="q9":    
        session['q8']=convertTofloat(answer)
    elif intent=="q10":    
        session['q9']=convertTofloat(answer)
    elif intent=="q11":    
        session['q10']=convertTofloat(answer)
    elif intent=="q12":    
        session['q11']=convertTofloat(answer)
    elif intent=="q13":    
        session['q12']=convertTofloat(answer)
    elif intent=="q14":    
        session['q13']=convertTofloat(answer)
    elif intent=="q15":    
        session['q14']=convertTofloat(answer)
    elif intent=="q16":    
        session['q15']=convertTofloat(answer)
    elif intent=="q17":    
        session['q16']=convertTofloat(answer)
    elif intent=="q18":    
        session['q17']=convertTofloat(answer)
    elif intent=="q19":    
        session['q18']=convertTofloat(answer)
    elif intent=="q20":    
        session['q19']=convertTofloat(answer)
    elif intent=="q21":    
        session['q20']=convertTofloat(answer)
    elif intent=="q22":    
        session['q21']=convertTofloat(answer)
    elif intent=="q23":    
        session['q22']=convertTofloat(answer)
    elif intent=="q24":    
        session['q23']=convertTofloat(answer)
    elif intent=="q25":    
        session['q24']=convertTofloat(answer)
    elif intent=="q26":    
        session['q25']=convertTofloat(answer)
    elif intent=="q27":    
        session['q26']=convertTofloat(answer)
    elif intent=="q28":    
        session['q27']=convertTofloat(answer)
    elif intent=="q29":    
        session['q28']=convertTofloat(answer)
    elif intent=="q30":    
        session['q29']=convertTofloat(answer)
    elif intent=="q31":    
        session['q30']=convertTofloat(answer)
    elif intent=="q32":    
        session['q31']=convertTofloat(answer)
    elif intent=="q33":    
        session['q32']=convertTofloat(answer)
    elif intent=="q34":    
        session['q33']=convertTofloat(answer)
    elif intent=="q35":    
        session['q34']=convertTofloat(answer)
    elif intent=="q36":    
        session['q35']=convertTofloat(answer)  
    elif intent=="finale":    
        session['q36']=convertTofloat(answer)
        hdType=calcType()
        session['hdType']=hdType    
        #print(session['q1'],session['q4'],session['q6'],session['q8'],session['q10'],session['q12'],session['q3'])
        
        
def chatControll(msg):
    text_to_be_analyzed = msg
    session_client = dialogflow.SessionsClient()
    session = session_client.session_path(DIALOGFLOW_PROJECT_ID, SESSION_ID)
    text_input = dialogflow.types.TextInput(text=text_to_be_analyzed, language_code=DIALOGFLOW_LANGUAGE_CODE)
    query_input = dialogflow.types.QueryInput(text=text_input)
    try:
        response = session_client.detect_intent(session=session, query_input=query_input)
    except InvalidArgument:
        raise
    
    validateAnswers(response.query_result.intent.display_name,response.query_result.query_text)
                
    print("Query text:", response.query_result.query_text)
    print("Detected intent:", response.query_result.intent.display_name)
    print("Fulfillment text:", response.query_result.fulfillment_text)
    
    return response

def calcType():    
    val1=session['q1']   #converting text into float
    val2=session['q2']
    val3=session['q3']
    val4=session['q4']
    val5=session['q5']
    val6=session['q6']
    val7=session['q7']
    val8=session['q8']
    val9=session['q9']
    val10=session['q10']
    val11=session['q11']
    val12=session['q12']
    val13=session['q13']
    val14=session['q14']
    val15=session['q15']
    val16=session['q16']
    val17=session['q17']
    val18=session['q18']
    val19=session['q19']
    val20=session['q20']
    val21=session['q21']
    val22=session['q22']
    val23=session['q23']
    val24=session['q24']
    val25=session['q25']
    val26=session['q26']
    val27=session['q27']
    val28=session['q28']
    val29=session['q29']
    val30=session['q30']
    val31=session['q31']
    val32=session['q32']
    val33=session['q33']
    val34=session['q34']
    val35=session['q35']
    val36=session['q36']   

    algorithm=joblib.load('Type_of_the_Disease.sav')
    #loading the trained algorithm
    result=algorithm.predict([[val1,val2,val3,val4,val5,val6,val7,val8,val9,val10,val11,val12,val13,val14,val15,val16,val17,val18,val19,val20,val21,val22,val23,val24,val25,val26,val27,val28,val29,val30,val31,val32,val33,val34,val35,val36]])

    return "Type of The Heart Disease:"+str(result[0])


#codes for Image Proccessing
def proccessImg(file):

   if request.method == 'POST':

        image = cv2.imread(file)
        img = cv2.resize(image, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)
        cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        filename = "{}.png".format('temp')
        cv2.imwrite(filename, img)
        text = pytesseract.image_to_string(cv2.imread(filename))
        print(text)
        os.remove(filename)

        TC=re.search(r'CHOLESTEROL(.*)mg/d',text).group(1)        
        TC=eval(re.findall("\d+\.\d+",TC)[0])

        HDL=re.search(r'(?:H.D.L|HDL)(.*?)mg/d', text).group(1)
        HDL=eval(re.findall("\d+\.\d+",HDL)[0])
        
        #checking canny
        edges = cv2.Canny(img,100,200)

        # show the output images
        cv2.imwrite("images/original.png", image)
        cv2.imwrite("images/preprocced.png", img)
        cv2.imwrite("images/canny.png", edges)
        
        return TC,HDL
    
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
    K.clear_session()
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



#Admin All

@app.route("/adminHome")
def adminHome():
    
   cursor = mydb.cursor(buffered=True)
   sql_select_query = "select * from doctor"
   cursor.execute(sql_select_query)
   record = cursor.fetchall()
   if record:
       return render_template('ViewDoctors.html',result=record)
   else:
       return render_template('ViewDoctors.html',result=[['There','is','no','data','to','retrive'],])
              
@app.route("/adminPatient")
def adminPatient():
    
   cursor = mydb.cursor(buffered=True)
   sql_select_query = "select * from patient"
   cursor.execute(sql_select_query)
   record = cursor.fetchall()
   if record:
       return render_template('ViewPatientAdmin.html',result=record)
   else:
       return render_template('ViewPatientAdmin.html',result=[['There','isnt','data','to','retrive'],])
    
    

@app.route('/disease') 
def disease():
    
   cursor = mydb.cursor(buffered=True)
   sql_select_query = "select * from disease"
   cursor.execute(sql_select_query)
   record = cursor.fetchall()
   if record:
       return render_template('disease.html',result=record)
   else:
       return render_template('disease.html',result=[['please','add','diseases'],])    


@app.route('/addDisease',methods=['GET', 'POST']) 
def addDisease():    
    
    if request.method == 'POST':
      did = request.form['did']
      name = request.form['name']
      symp = request.form['symp']
              
    cursor = mydb.cursor(buffered=True)
    sql = """INSERT INTO disease (id, name,symptomps) VALUES (%s, %s ,%s)"""
    val = (did,name,symp)
    cursor.execute(sql, val)
    mydb.commit() 
    return redirect(url_for('disease'));

@app.route('/updateDisease',methods=['GET', 'POST']) 
def updateDisease():    
    
    if request.method == 'POST':
      did = request.form['did']
      name = request.form['name']
      symp = request.form['symp']
       
              
    cursor = mydb.cursor(buffered=True)
    sql = """UPDATE disease SET name =%s,symptomps=%s WHERE id=%s """
    val = (name,symp,did)
    cursor.execute(sql, val)
    mydb.commit()
    
    return redirect(url_for('disease'))

@app.route('/deleteDisease',methods=['GET', 'POST']) 
def deleteDisease():    
    
    if request.method == 'POST':
      did = request.form['did']       
              
    cursor = mydb.cursor(buffered=True)
    sql = " DELETE FROM disease WHERE id = %s "
    val = (did,)
    cursor.execute(sql, val)
    mydb.commit()
    
    return redirect(url_for('disease'))

@app.route('/manPatient') 
def manPatient():
    
   cursor = mydb.cursor(buffered=True)
   sql_select_query = "select * from patient"
   cursor.execute(sql_select_query)
   record = cursor.fetchall()
   if record:
       return render_template('ViewPatient.html',result=record)
   else:
       return render_template('ViewPatient.html',result={{'test','test','test','test','test','test'},})





#Doctor ALL

@app.route("/doctorHome")
def doctorHome():
    
   cursor = mydb.cursor(buffered=True)
   sql_select_query = "select * from patient"
   cursor.execute(sql_select_query)
   record = cursor.fetchall()
   if record:
       return render_template('ViewPatient.html',result=record)
   else:
       return render_template('ViewPatient.html',result={{'test','test','test','test','test','test'},})


@app.route('/viewPatients') 
def viewPatients():    

    return render_template('ViewPatient.html')

@app.route('/viewReports') 
def viewReports():    

    return render_template('ViewPatient.html')




#Patient routes

@app.route("/patientHome")
def patientHome():
    
    cursor = mydb.cursor(buffered=True)
    sql_select_query = "SELECT * FROM disease"
    cursor.execute(sql_select_query)
    record = cursor.fetchall()
    if record:
       return render_template('Dtree.html',result=record)
    else:
       return render_template('Dtree.html',result={{'test','test','test','test','test'},})

@app.route('/diseasePatient') 
def diseasePatient():    

    return render_template('disease.html')

@app.route('/exercises') 
def exercises():    

    return render_template('exercises.html',result ="0.png")

@app.route('/risk') 
def risk():    

    return render_template("risk.html",result="result") 


@app.route('/chat') 
def chat():    
    name=session['name'];
    session['hdType']="no"
    return render_template('chatbot.html',result=name)

@app.route('/meal') 
def meal():    

    return render_template('meal_plan.html',result ="0.png")






#Login,Register and Logout

@app.route('/logout') 
def logout():
    
    session['Uid']="null"
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
   if record:
       for x in record:
           if x is None:
               return redirect(url_for('logAdmin')) 
           else:    
               session['Uid']=record[0][0] 
               return redirect(url_for('adminHome'))  
   else:
       return redirect(url_for('logAdmin'))
   
    
@app.route('/loginPatient',methods = ['POST', 'GET'])
def loginPatient():

   if request.method == 'POST':
      username = request.form['uname']
      pw = request.form['pwd']
              
   cursor = mydb.cursor(buffered=True)
   sql_select_query = "select * from patient where email = %s and Password = %s"
   cursor.execute(sql_select_query, (username,pw))
   record = cursor.fetchall()
   if record:
       for x in record:
           if x is None:
               return redirect(url_for('logPat')) 
           else:    
               session['Uid']=record[0][0]
               session['name']=record[0][2]
               #print(record[0][0])
               return redirect(url_for('patientHome'))       
   else:
       return redirect(url_for('logPat'))

 
@app.route('/loginDoctor',methods = ['POST', 'GET'])
def loginDoctor():

   if request.method == 'POST':
      username = request.form['uname']
      pw = request.form['pwd']
              
   cursor = mydb.cursor(buffered=True)
   sql_select_query = "select * from doctor where drn = %s and Password = %s"
   cursor.execute(sql_select_query, (username,pw))
   record = cursor.fetchall()
   if record:
       for x in record:
           if x is None:
               return redirect(url_for('logDoc')) 
           else:    
               session['Uid']=record[0][0]
               session['name']=record[0][2]
               return redirect(url_for('doctorHome')) 
       
   else:
        return redirect(url_for('logDoc'))

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




#Patient fuctions

@app.route('/rate') 
def rate():    

    return render_template('rating.html')

@app.route('/viewRates') 
def viewRates():    
   cursor = mydb.cursor(buffered=True)
   sql_select_query = "select * from rate"
   cursor.execute(sql_select_query)
   record = cursor.fetchall()
   if record:
       return render_template('customerRatings.html',result=record)

@app.route('/saveRate',methods = ['POST', 'GET']) 
def saveRate():

   if request.method == 'POST':
      r1 = request.form['r1']
      r2 = request.form['r2']
      r3 = request.form['r3']
   email=session['Uid']   

   cursor = mydb.cursor(buffered=True)
   sql_select_query = "select * from rate where email=%s"
   cursor.execute(sql_select_query,(email,))
   record = cursor.fetchall()
   if record:
        sql = """UPDATE rate SET r1=%s,r2=%s,r3=%s WHERE email=%s """
        val = (r1,r2,r3,email)
        cursor.execute(sql, val)
        mydb.commit()
        return redirect(url_for('viewRates'))
   else:
        sql = """INSERT INTO rate (email,r1,r2,r3) VALUES (%s, %s ,%s ,%s)"""
        val = (email,r1,r2,r3)
        cursor.execute(sql, val)
        mydb.commit() 
        return redirect(url_for('viewRates'))    


@app.route('/kalpana',methods = ['POST', 'GET'])
def kalpana():
    
    model=load_model()
    
    if request.method == 'POST':
      age = int(request.form['age'])
      gender = int(request.form['gender'])
      med = int(request.form['blood'])
      dia = int(request.form['diabet'])
      smoke = int(request.form['Smoke'])
      sbp = int(request.form['sbp'])
      img=request.files['image']
      img.save('output.PNG')
    tc,hdl=proccessImg('output.PNG')
    
    print(tc," and ",hdl)
    test_data=[gender,age,tc,hdl,sbp,smoke,med,dia]
    
    test_data=scaler_x.transform([test_data])
    ped_result=model.predict(test_data)
    K.clear_session()
    
    value=ped_result[0][0]
    risk=value*100
    return render_template("risk.html",result=int(risk)) 

#meal plan
@app.route('/predict_mealPlan',methods=['GET','POST']) 
def predict_mealPlan():    

    data=request.form
    val1=eval(data['gender'])   #converting text into float
    val2=eval(data['age'])
    val4=eval(data['risk'])
    hight=eval(data['hh'])
    weight=eval(data['ww'])
    
    val3=weight/hight
    
    algorithm=joblib.load('Meal_Plan_SVM_model.sav')
    #loading the trained algorithm
    result=algorithm.predict([[val1,val2,val3,val4]])

    #print(val1,val2,val3,val4,result)

    ex=str(int(result[0]))
    img=ex+".png"
    return render_template("meal_plan.html",result=img) 

#exersise 
@app.route('/predict_exercises',methods=['GET','POST']) 
def predict_exercises():    

    data=request.form
    val01=eval(data['gender1'])   #converting text into float
    val02=eval(data['age1'])
    val03=eval(data['risk1'])

    algorithm=joblib.load('Exercises_SVM_model.sav')
    #loading the trained algorithm
    result=algorithm.predict([[val01,val02,val03]])

    #print(val1,val2,val3,val4,result)

    ex=str(int(result[0]))
    img=ex+".png"
    return render_template("exercises.html",result =img)

#chatbot
@app.route("/ask", methods=['POST','GET'])
def ask():
        
    message = str(request.form['chatmessage'])
    response=chatControll(message)
    if session['hdType'] == "no":
        return jsonify({'status':'OK','answer':response.query_result.fulfillment_text})
    else:
        hdType=session['hdType']
        return jsonify({'status':'OK','answer':hdType})
    
        
if __name__ == "__main__":
	app.run(debug=True, use_reloader=False)
   


