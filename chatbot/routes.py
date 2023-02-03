from chatbot import app
from flask import render_template,flash, request
from chatbot.forms import chatbotform
from chatbot.__init__ import model,words,classes,intents

import nltk
import pickle
import json
import numpy as np
from keras.models import Sequential,load_model
import random
from datetime import datetime
import pytz
import requests
import os
import billboard
import time
from pygame import mixer
import COVID19Py

from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()


#Predict
def clean_up(sentence):
    sentence_words=nltk.word_tokenize(sentence)
    sentence_words=[ lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def create_bow(sentence,words):
    sentence_words=clean_up(sentence)
    bag=list(np.zeros(len(words)))

    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence,model):
    p=create_bow(sentence,words)
    res=model.predict(np.array([p]))[0]
    threshold=0.8
    results=[[i,r] for i,r in enumerate(res) if r>threshold]
    results.sort(key=lambda x: x[1],reverse=True)
    return_list=[]

    for result in results:
        return_list.append({'intent':classes[result[0]],'prob':str(result[1])})
    return return_list

def get_response(return_list,intents_json,text):

    if len(return_list)==0:
        tag='noanswer'
    else:
        tag=return_list[0]['intent']
    if tag=='datetime':
        x=''
        tz = pytz.timezone('Asia/Kolkata')
        dt=datetime.now(tz)
        x+=str(dt.strftime("%A"))+' '
        x+=str(dt.strftime("%d %B %Y"))+' '
        x+=str(dt.strftime("%H:%M:%S"))
        return x,'datetime'

    if tag=='timer':
        x=text.split(':')[1].strip()
        time.sleep(float(x)*60)
        x='Timer ringing...'
        return x,'timer'

    list_of_intents= intents_json['intents']
    for i in list_of_intents:
        if tag==i['tag'] :
            result= random.choice(i['responses'])
    return result,tag

def response(text):
    return_list=predict_class(text,model)
    response,_=get_response(return_list,intents,text)
    return response



@app.route('/',methods=['GET','POST'])
def yo():
    return render_template('main.html')

@app.route('/chat',methods=['GET','POST'])
def home():
    return render_template('index.html')

@app.route("/get")
def chatbot():
    userText = request.args.get('msg')
    resp=response(userText)
    return resp
