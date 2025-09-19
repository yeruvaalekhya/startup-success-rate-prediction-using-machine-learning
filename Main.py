from flask import Flask, render_template, request, redirect, url_for, session,send_from_directory
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd

app = Flask(__name__)

app.secret_key = 'welcome'
global user

dataset = pd.read_csv("Dataset/startup_data.csv")
data = dataset[['relationships', 'funding_rounds', 'funding_total_usd', 'milestones', 'avg_participants', 'status']]
le = LabelEncoder()
data['status'] = pd.Series(le.fit_transform(data['status'].astype(str)))#encode all str columns to numeric
Y = data['status']
X = data.values[:,0:data.shape[1]-1]
#normalizing training features
scaler = StandardScaler()
X = scaler.fit_transform(X)
data = np.load("model/data.npy", allow_pickle=True)
X_train, X_test, y_train, y_test = data
rf_cls = RandomForestClassifier(bootstrap=False, max_depth=12, min_samples_leaf=2, min_samples_split=2, n_estimators=100)
rf_cls.fit(X_train, y_train)

@app.route('/PredictAction', methods=['GET', 'POST'])
def PredictAction():
    if request.method == 'POST':
        global rf_cls, scaler
        relation = request.form['t1']
        funding = request.form['t2']
        usd = request.form['t3']
        milestone = request.form['t4']
        participant = request.form['t5']
        data = []
        data.append([float(relation.strip()), float(funding.strip()), float(usd.strip()), float(milestone.strip()), float(participant.strip())])
        data = np.asarray(data)
        data = scaler.transform(data)
        predict = rf_cls.predict(data)[0]
        result = '<font size="3" color="green">Your startup will be successfull</font>' 
        if predict == 1:
            result = '<font size="3" color="red">Your startup will be Failed</font>'
        suggestion = ""
        if float(relation.strip()) <= 0 and float(funding.strip()) <= 0 and float(usd.strip()) <= 0 and float(milestone.strip()) <= 0 and float(participant.strip()) <= 0:
            suggestion = 'Please enter valid data'
        elif float(milestone.strip()) < 4:
            suggestion = 'Our prediction says that your start-up might failed as Milestones are less.<br/>Please try to increase milestones then you will definitely succeed.'
        elif float(relation.strip()) < 4:
            suggestion = 'Our prediction says that your start-up might failed as Relationships are less.<br/>Please try to increase relationships then you will definitely succeed.'
        elif float(funding.strip()) < 4:
            suggestion = 'Our prediction says that your start-up might failed as Funding Rounds are less.<br/>Please try to increase Funding Rounds then you will definitely succeed.'
        elif float(usd.strip()) < 4:
            suggestion = 'Our prediction says that your start-up might failed as Total Amount is not sufficient.<br/>Please try to increase Total Amount then you will definitely succeed.'
        else:
            suggestion = 'Our prediction says that your start-up will hopefully a success.'
        values = "Relationships = "+relation+"<br/>"
        values += "Funding Rounds = "+funding+"<br/>"
        values += "Average Funding USD = "+usd+"<br/>"
        values += "Milestones = "+milestone+"<br/>"
        values += "Average Partipicant = "+participant+"<br/>"
        return render_template('PredictSuccess.html', data=values+"<br/>"+result+"<br/>Suggestion = "+suggestion)

@app.route('/PredictSuccess', methods=['GET', 'POST'])
def PredictSuccess():
    return render_template('PredictSuccess.html', data='')

@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html', data='')

@app.route('/Logout')
def Logout():
    return render_template('index.html', data='')

if __name__ == '__main__':
    app.run()










