import streamlit as st
import helper
import pickle

model = pickle.load(open('model.pkl','rb'))

st.header('Duplicate Quora Question Pairs')

q1 = st.text_input('Question 1')
q2 = st.text_input('Question 2')

if st.button('Predict'):
    query = helper.query_point_creator(q1,q2)
    result = model.predict(query)[0]

    if result == 1:
        st.header('Duplicate')
    else:
        st.header('Non-duplicate')