from PIL import Image
import pytesseract
import cv2
import os

def proccessImg():
    pytesseract.pytesseract.tesseract_cmd='C://Program Files/Tesseract-OCR/tesseract.exe'

    image = cv2.imread('images/5.PNG')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #gray = cv2.medianBlur(gray, 3)

    filename = "{}.png".format(' temp')
    cv2.imwrite(filename, gray)
    text = pytesseract.image_to_string(cv2.imread(filename))
    os.remove(filename)

    print(text)

    import re

    print(text)
    print('====EXTRACTED PARAMETERS====')
    TC=re.search(r'CHOLESTEROL(.*?)mg/dL', text).group(1)
    TC=eval(re.findall("\d+\.\d+",TC)[0])

    HDL=re.search(r'H.D.L(.*?)mg/dL', text).group(1)
    HDL=eval(re.findall("\d+\.\d+",HDL)[0])

    print('TC:',TC)
    print('HDL:',HDL)
    
    return TC, HDL;
