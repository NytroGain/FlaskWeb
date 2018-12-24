from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import pickle
import json

import pandas as pd
import numpy as np
from sklearn import tree

from flask import send_from_directory, Response

from keras.models import load_model
from keras import backend as K

app = Flask(__name__)

@app.route('/')
def upload_file1():
    return render_template('home.html')
#--------------------------------------------------------------------------------------------------#
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
       df = pd.read_csv(request.files.get('file'), sep=',',header=0, encoding='TIS-620')
       hpno = df['Account']
       df = df.drop('Account', axis=1)
       
       #------------------getDUMMY สำหรับเทียบให้เหมือนกับโมเดล----------------------------#
       ord1 = pd.read_csv('afterFeatureSelectionCSV.csv',sep=',',header=0, encoding='TIS-620')
       ord1 = ord1.drop('ACI', axis=1)
       for column in ord1.select_dtypes(include=[np.object]).columns:
           df[column] = df[column].astype('category', categories = ord1[column].unique())
       #---------------------------------------------------------------------------------#
       oh = pd.get_dummies(df)                                  #แปลงค่าให้อยู่ในรูปแบบ One Hot Encoding
       md = pickle.load(open('dtree_model_pickleMD10.p','rb'))  #เรียกใช้โมเดลDecision Tree
       y_pred = md.predict(oh)
       y_prob = md.predict_proba(oh)
       Most_prob = []
       for v in range(len(y_prob)):
           if y_prob[v][0] > y_prob[v][1]:
               Most_prob.append(y_prob[v][0])
           else: Most_prob.append(y_prob[v][1])
       AnsYES = []      #------เก็บคำตอบYES ปล.ไม่ได้ใช้แล้ว
       AccountYes = []  #------เก็บคำตอบรหัสACCOUNT
       proYes = []      #------เก็บคำตอบความน่าจะเป็น
       #--------คัดเฉพาะคำตอบ YES-------------------------------#
       for x in range(len(y_pred)):
           if y_pred[x] == "YES":
               AnsYES.append(y_pred[x])
               AccountYes.append(hpno[x])
               proYes.append(Most_prob[x])
       #p = pd.DataFrame(y_pred)
       #p=p.to_csv('onlyYES.csv')
       #--------------------------------------------------------------------#

       return render_template('result.html',valuxs=AnsYES, Acc=AccountYes, proB=proYes)     #แสดงผลลัพธ์
#--------------------------------------------------------------------------------------------------
@app.route('/uploaderzwei', methods = ['GET', 'POST'])
def upload_filezwei():
    if request.method == 'POST':
       df = pd.read_csv(request.files.get('file'), sep=',',header=0, encoding='TIS-620')
       hpno = df['Account']
       df = df.drop('Account', axis=1)
       
       #------------------getDUMMY สำหรับเทียบให้เหมือนกับโมเดล----------------------------#
       ord1 = pd.read_csv('afterFeatureSelectionCSV.csv',sep=',',header=0, encoding='TIS-620')
       ord1 = ord1.drop('ACI', axis=1)
       for column in ord1.select_dtypes(include=[np.object]).columns:
           df[column] = df[column].astype('category', categories = ord1[column].unique())
       #---------------------------------------------------------------------------------#
       oh = pd.get_dummies(df)
       md = load_model('TestModelT1000.h5')  #เรียกใช้โมเดลNeural Netwok
       y_pred = md.predict_classes(oh)
       y_proba = md.predict_proba(oh)
       K.clear_session()
       
       #p = pd.DataFrame(y_pred)
       #p=p.to_csv('NNAns300.csv', index=False)

       AnsYES = []      #------เก็บคำตอบYES ปล.ไม่ได้ใช้แล้ว
       AccountYes = []  #------เก็บคำตอบรหัสACCOUNT
       proYes = []      #------เก็บคำตอบความน่าจะเป็น
       #--------คัดเฉพาะคำตอบ YES-------------------------------#
       for x in range(len(y_pred)):
           if y_pred[x][0] == 1:
               AnsYES.append(y_pred[x][0])
               AccountYes.append(hpno[x])
               proYes.append(y_proba[x][0])


       return render_template('result.html', valuxs=AnsYES, Acc=AccountYes, proB=proYes)

#--------------------------------------------------------------------------------------------------


@app.route("/Log")
def showlog():
    return render_template('Log.html')

@app.route("/Preprocess")
def showPre():
    return render_template('Preprocess.html')
'''
@app.route("/getPlotCSV", methods=['POST', 'GET'])
def getPlotCSV():
    if request.method == 'POST':
        csv = request.args.get('name001')
        return Response(
            csv,
            mimetype="text/csv",
            headers={"Content-disposition":
                    "attachment; filename=Result.csv"})
'''
if __name__ == '__main__':
   app.run(debug = True)