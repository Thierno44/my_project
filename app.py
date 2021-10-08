import streamlit as st
import os
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


uploaded_file = st.file_uploader("Selectionner des données de test",type=['csv'])

if uploaded_file is not None:
   file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
   st.write(file_details)
   df = pd.read_csv('Dataset.csv', sep=";", encoding ="latin-1")
   df = df.drop(["TREATY ID", "CEDANTID", "PRODUCT ID"], axis = 1)
   df['PRIMES'] = pd.to_numeric(df['PRIMES'], errors='ignore')
   
   df_test=pd.read_csv(uploaded_file, sep=";", encoding ="latin-1")
   df_test = df_test.drop(["TREATY ID", "CEDANTID", "PRODUCT ID"], axis = 1)
   df_test['PRIMES'] = pd.to_numeric(df_test['PRIMES'], errors='ignore')
   
   X, y = df.drop("TARGET", axis = 1), df['TARGET']
   
   encod_test = OneHotEncoder(drop='first', sparse=False)
   encod_test.fit(df_test)
   encod = OneHotEncoder (drop='first', sparse=False)
   encod.fit(X)
   X= encod.transform (X)
   
   label = LabelEncoder()
   y = label.fit_transform(y)
   
   #st.write("la taille du jeu de données", X.shape, y.shape)
   
   #st.write("taille du df_testset", df_test.shape)
   
   #st.write("taille df_testset apres suppression", df_test.shape)
   
   
   #st.write("taille df_testset apres instanciation", df_test.shape)
   
   
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .33, random_state = 1234)
   
   model = LogisticRegression(C=1.0,max_iter=100)
   model.fit(X_train, y_train)
   y_pred = model.predict(X_test)
   #model = 'finalized_model.sav'
   #loaded_model = pickle.load(open(model, 'rb'))
   
   
   #st.write(df_test.shape)
   
   
   #loaded_model = pickle.load(open(model, 'rb'))
   
   df_test = encod_test.transform (df_test)
   #y_pred = loaded_model.predict(df_test) 
   
   score = metrics.accuracy_score (y_test, y_pred)
   st.write("Le score du modèle est de %", score)
   
   y_prediction = model.predict(df_test)
   
   df_pred = encod.inverse_transform(df_test)
   
   df_pred = pd.df_testFrame(df_pred, 
             columns = ["TREATY NAME","ZONE", "COUNTRY", "CEDANT", "BROKER", "FAMILY OF BUSINESS","PRODUCT", 
                        "METHOD", "SECTION", "EXERCISE", "BILAN", "PRIMES"])
   predict = df_pred
   predict['Resultat']= label.inverse_transform(y_prediction)
   st.write(predict.to_csv("predictions.csv"))  
      




