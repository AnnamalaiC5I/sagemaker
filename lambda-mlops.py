import os
import json
import boto3
import pickle
import sklearn
import warnings
warnings.simplefilter("ignore")

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')
bucket = "sagemaker-project-p-p1wujcenlugu"
s3 = boto3.resource('s3')

def lambda_handler(event, context):
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='text/csv',
                                       Body=event)
    result = json.loads(response['Body'].read().decode())
   
    output_dict = {0:'setosa',1:'versicolor',2:'virginica'}
    
    ans = output_dict[result]
    
    return ans