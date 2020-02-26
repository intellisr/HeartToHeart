from flask import Flask,render_template,request
import joblib

#from PIL import Image
#import pytesseract
#import cv2
#import os
#import re

#pytesseract.pytesseract.tesseract_cmd='C://Program Files/Tesseract-OCR/tesseract.exe'

app=Flask(__name__) #empty web app

@app.route('/') #endpoint
def index():    #function of thee endpoint

    return render_template('home.html')

@app.route('/excersice') 
def disease_type():    

    return render_template('excersice.html')

@app.route('/image_upload') 
def image_upload():    

    return render_template('image_upload.html')

@app.route('/predict_exercise',methods=['GET','POST']) 
def predict_exercise():    

    data=request.form
    val1=eval(data['v1'])   #converting text into float
    val2=eval(data['v2'])
    val3=eval(data['v3'])
    val4=eval(data['v4'])

    algorithm=joblib.load('Meal_Plan_SVM_model.sav')
    #loading the trained algorithm
    result=algorithm.predict([[val1,val2,val3,val4]])

    #print(val1,val2,val3,val4,result)

    return "PREDICTED MEAL PLAN:"+str(int(result[0]))

"""
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():

   if request.method == 'POST':

        f = request.files['file']
        f.save('output.PNG')

        image = cv2.imread('output.PNG')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #gray = cv2.medianBlur(gray, 3)

        filename = "{}.png".format(' temp')
        cv2.imwrite(filename, gray)
        text = pytesseract.image_to_string(cv2.imread(filename))
        os.remove(filename)

        TC=re.search(r'CHOLESTEROL(.*?)mg/dL', text).group(1)
        TC=eval(re.findall("\d+\.\d+",TC)[0])

        HDL=re.search(r'H.D.L(.*?)mg/dL', text).group(1)
        HDL=eval(re.findall("\d+\.\d+",HDL)[0])

        # show the output images
        cv2.imwrite("static/output-1.png", image)
        cv2.imwrite("static/output-2.png", gray)
    
        return render_template('image_result.html',result='TC:'+str(TC)+'     HDL:'+str(HDL))
"""
app.run(debug=True)
