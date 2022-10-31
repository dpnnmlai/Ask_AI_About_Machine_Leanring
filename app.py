from cProfile import label
from unittest import result
from matplotlib.pyplot import text
import streamlit as st
import os 



st.markdown("<h2 style='text-align: center; color: white;'>🤖Ask Me about Machine Learning🤖</h2>", unsafe_allow_html=True)
st.markdown('')
st.markdown('')


st.session_state['new']=True

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

form = st.form(key='my_form')

# Creating the Pipeline 

nlp = pipeline('question-answering', model='deepset/roberta-base-squad2', tokenizer='deepset/roberta-base-squad2')

text = form.text_area('ML Stuff to Learn 🖥️🤖')

submit_button = form.form_submit_button(label='Learn This')

st.markdown('---')
ques=st.text_input('Ask Me Anything You Want About Machine Learning')

ques_dist = {
    'question':ques,
    'context':text
}

butt = st.button('Ask')

if butt ==True:
    results = nlp(ques_dist)
    st.markdown('---')
    st.subheader('Here is Your Answer')
    st.success(results['answer'])
    st.balloons()