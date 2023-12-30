import streamlit as st
import pickle
import numpy as np
import sklearn
st.title('Titanic prediction')
pipe=pickle.load(open('pipe.pkl','rb'))
df=pickle.load(open('df .pkl','rb'))
pclass=st.selectbox('Pclass',df['pclass'].unique())
gender=st.selectbox('Gender',df['sex'].unique())
age=st.selectbox('Age',df['age'].unique())
parch=st.selectbox('Parch',df['parch'].unique())
embar=st.selectbox('embarked',df['embarked'].unique())
clas=st.selectbox('class',df['class'].unique())
who=st.selectbox('who',df['who'].unique())
if st.button('Predict'):
    query=np.array([pclass,gender,age,parch,embar,clas,who])
    query=query.reshape(1,7)
    st.title(pipe.predict(query))


    