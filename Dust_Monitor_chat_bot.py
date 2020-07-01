import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import re
from tensorflow.keras.models import load_model
#from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents_dust_monitor.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=True)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.75
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# def getResponse(ints, intents_json):
#     tag = ints[0]['intent']
#     list_of_intents = intents_json['intents']
# # 	 for regex in QRregexList:
# #     check = re.search(list_of_intents,tag, re.IGNORECASE)
#    # if check:
#     for i in list_of_intents:
#         check=re.search(i['tag'],tag,re.IGNORECASE)
#         if check:
#             result = random.choice(i['responses'])
#             break

#         # else:
#         #    result="Sorry, can't understand you Please give me more info"


#     return result

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    print(ints)
    if ints==[]:
       res="Sorry,Not able to understand your question can you please check any spelling mistakes or provide me more relevent keywords and information"
    else:
     res = getResponse(ints, intents)
    return res



import streamlit as st


st.write("""
## This is Simple Dust Monitor related Question answer web App

Here you can ask questions  related to dust monitor components,commissioning,tools,erection,calibration,Alignment ,reset,Dust factor ,errors Etc.
""")

st.subheader('Enter your Dust Monitor related Question here:')
#sentence = st.text_input('Enter your dust monitor related question here:',key="t1") 
sentence = st.text_input(" ") 


if sentence:
    st.subheader('Answer:')
    st.write(chatbot_response(sentence))
    #sentence=st.text_input('Input your sentence here:', key="t2" ) 

    




