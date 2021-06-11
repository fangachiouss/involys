#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 03:57:36 2020

@author: root
"""

from pycaret.regression import load_model,predict_model
import streamlit as  st
import pandas as pd
import numpy as np
import datetime as dt
from pickle import  *
regression =load_model('deployment_28042021')


def predict(model,input_df):
    predictions_df=predict_model(estimator=model,data=input_df)
    predictions=predictions_df
    return predictions
def run(): 
    add_selectbox=st.sidebar.selectbox("how would you to predict",("online","fichier","graph"))
    st.sidebar.success('https://www.pycaret.org')
    if(add_selectbox=='online'):
        f1 = open ("column","rb")
        column=load(f1)
        type_depense=st.selectbox("Type dépense",np.array(column["Type dépense"]))
        montant_depense=st.number_input("Montant Dépense")
        Service_demandeur=st.selectbox("Service demandeur",np.array(column["Service demandeur"]))
        Date_signature=st.date_input('Date signature')
        delai_execution=st.number_input("Délai exécution")
        Fournisseur=st.selectbox("Fournisseur",np.array(column["Fournisseur"]))
        date1=st.date_input('Date validation facture') 
        Montant_facture_HT=st.number_input('Montant facture HT')
        input_dict={"Type dépense":type_depense,"Montant Dépense":montant_depense,"Service demandeur":Service_demandeur,"Date signature":Date_signature,"Délai exécution":delai_execution,"Fournisseur":Fournisseur,"Date validation facture":date1,"Montant facture HT":Montant_facture_HT}
        data_predict=pd.DataFrame([input_dict])
        data_predict['Date validation facture']=pd.to_datetime(data_predict['Date validation facture'])
        data_predict['Date signature']=pd.to_datetime(data_predict['Date signature'])
        #data_predict['Date validation facture']=data_predict['Date validation facture'].map(dt.datetime.toordinal)
        #data_predict['Date signature']=data_predict['Date signature'].map(dt.datetime.toordinal)
        for col in data_predict.select_dtypes(include=['object']):
            f = open (col,"rb")
            num=load(f)
            data_predict[col]=data_predict[col].map(num)
        for col in data_predict.select_dtypes(include=['datetime64']):
            data_predict[col]=data_predict[col].map(dt.datetime.toordinal)            
        if(st.button("predict")):
            output=predict(model=regression,input_df=data_predict)
            st.write(int(output['Label']))
    if(add_selectbox=='fichier'):  
        file_upl=st.file_uploader("Upload excel file for predictions" ,type=["xlsx"])
        if(file_upl is not None):
            data=pd.read_excel(file_upl)
            data_predict1=data.copy()
            for col in data_predict1.select_dtypes(include=['object']):
                f = open (col,"rb")
                num=load(f)
                data_predict1[col]=data_predict1[col].map(num)
            for col in data_predict1.select_dtypes(include=['datetime64']):
                data_predict1[col]=data_predict1[col].map(dt.datetime.toordinal)
            predictions=predict_model(estimator=regression,data=data_predict1)
            data['Label']=predictions['Label'].astype(int)
            st.write(data)
    if(add_selectbox=='graph'):
        if(st.button("residual")):
            st.image("Residuals.png")
        if(st.button("feature importance")):
            st.image("Feature Importance.png")
        if(st.button("feature selection")):
            st.image("Feature Selection.png")
        if(st.button("error")):
            st.image("Prediction Error.png")
        if(st.button("interpret")):
            st.image("SHAP summary.png")





run()      
