import streamlit as st
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
st.title('Survival chance of titanic using ANN')
pclass=st.slider('Enter Passenger Class',1,3)
sex = st.selectbox("Enter sex",['male','female'])
SibSp = st.slider("Enter sibling spouse:",1,8)
Parch = st.slider('Enter Parents Children',0,6)
fare = st.number_input('Enter Fare')
Embarked = st.selectbox(" select the city for embarkment",['Southampton','Queenstown','Cherbourg'])

with open('label.pkl','rb') as f:
    label = pickle.load(f)

with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)
with open('onehot.pkl','rb') as f:
    onehot = pickle.load(f)
model = load_model('model.h5')
user = {'Pclass':pclass,'Sex':sex,'SibSp':SibSp,'Parch':Parch,'Fare':fare,'Embarked':Embarked}

user = pd.DataFrame(user, index=[0])

user['Sex'] = label.transform(user['Sex'])
embarked = onehot.transform(user[['Embarked']])
embarked = pd.DataFrame(embarked,columns=onehot.get_feature_names_out())
user = pd.concat([user.drop(columns=['Embarked']),embarked],axis=1)
user[['Pclass','SibSp','Parch','Fare']] = scaler.transform(user[['Pclass','SibSp','Parch','Fare']])
if st.button('enter'):
    val = model.predict(user)
    if val[0][0]>0.5:
        st.write(f'The Passenger Survive: {val[0][0]}')
    else:
        st.write(f'The Passenger does not Survive: {val[0][0]}')