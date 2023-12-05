import streamlit as st
import pandas as pd
import pickle
import numpy as np
import requests
import json
import boto3

import io
from io import StringIO

# filename = 'finalized_model.sav'
# model = pickle.load(open(filename, 'rb'))

st.title('Iris Flower Prediction Model')

pl = st.number_input(label = 'Petal length',min_value=0.0,max_value=6.0,step=0.1)

pw = st.number_input(label = 'Petal Width',min_value=0.0,max_value=6.0,step=0.1)

sl = st.number_input(label = 'Sepal Length',min_value=0.0,max_value=6.0,step=0.1)

sw = st.number_input(label = 'Sepal Width',min_value=0.0,max_value=6.0,step=0.1)

#lis = np.array([sl,sw,pl,pw]).reshape(1,4)

df = pd.DataFrame({'sepal_length' : [sl], 'sepal_width' : [sw],'petal_length':[pl],'petal_width':[pw]})

csv_file = io.StringIO()
# by default sagemaker expects comma seperated
df.to_csv(csv_file, sep=",", header=False, index=False)
my_payload_as_csv = csv_file.getvalue()




submit = st.button('submit')

if submit:
    #  out = model.predict(lis)
    #  st.success('The predicted output is {}'.format(out[0]))
    client = boto3.client('sagemaker-runtime')
    response = client.invoke_endpoint(EndpointName="project-x1-staging",Body= my_payload_as_csv,ContentType = 'text/csv')

    response_body = response['Body']
    ans = response_body.read().decode()

    ans_dict = {'0.0':'iris-setosa','1.0':'iris-versicolor','2.0':'iris-virginica','':'wrong-response'}

    j = ans_dict[ans]

    st.success('The predicted output is {}'.format(j))
