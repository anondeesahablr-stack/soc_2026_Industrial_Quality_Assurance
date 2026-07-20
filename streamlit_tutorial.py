import streamlit as st
st.title('HelloWorld')
st.text('Welcome to an AI-powered Industrial Quality Assurance system')
st.write('Upload a steel surface image')
col1, col2 = st.columns(2)
with col1:
    st.header('Image Upload')

with col2:
    st.header('LLM')
    