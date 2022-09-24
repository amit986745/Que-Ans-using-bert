import streamlit as st
import helper
import pickle
model=pickle.load(open(r"C:\Users\AMIT PAREEK\Documents\mtech 2nd sem\nlp\assignment\model.pkl","rb"))
st.header('Duplicate question pairs')
q1=st.text_input('Enter question 1')
q2=st.text_input('Enter question 2')

if st.button('Find'):
    query=helper.query_point_creator(q1,q2)
    result=model.predict(query)[0]
    if result:
        st.header('Duplicate')
    else:
        st.header('Not duplicate')