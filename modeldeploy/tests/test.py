import argparse
import json
import logging
import os
import pandas as pd
from sklearn.metrics import accuracy_score
import io

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
sm_client = boto3.client("sagemaker")


def invoke_endpoint(endpoint_name):
    """
    Add custom logic here to invoke the endpoint and validate reponse
    """
    fn = 'IRIS.csv'
    s3 = boto3.resource("s3",aws_access_key_id='AKIAUJKJ5ZIQJK6XUY6X', 
                      aws_secret_access_key='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', 
                      region_name='ap-south-1')
   
    s3.Bucket('iris-data-model').download_file('IRIS.csv', fn) 

    data = pd.read_csv(fn)
    os.unlink(fn)
    
    features = data.drop(['species'],axis=1)
    target = data['species']
    
    re_ans_dict = {'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}

    ans1=list()
    for num in target:
         answers = re_ans_dict[num]
         ans1.append(answers)

    csv_file = io.StringIO()
    # by default sagemaker expects comma seperated
    features.to_csv(csv_file, sep=",", header=False, index=False)
    my_payload_as_csv = csv_file.getvalue()

    client = boto3.client('sagemaker-runtime')
    response = client.invoke_endpoint(
                EndpointName=endpoint_name,
                Body= my_payload_as_csv,
                ContentType = 'text/csv')
    
    response_body = response['Body']
    
    ans = response_body.read().decode()
    
    numbers = [int(float(num)) for num in ans.split(',')]
    
    acc = accuracy_score(ans1,numbers)
    
    return {"endpoint_name": endpoint_name,"accuracy_score":acc,"success": True}


def test_endpoint(endpoint_name):
    """
    Describe the endpoint and ensure InSerivce, then invoke endpoint.  Raises exception on error.
    """
    error_message = None
    try:
        # Ensure endpoint is in service
        response = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]
        if status != "InService":
            error_message = f"SageMaker endpoint: {endpoint_name} status: {status} not InService"
            logger.error(error_message)
            raise Exception(error_message)

        # Output if endpoint has data capture enbaled
        endpoint_config_name = response["EndpointConfigName"]
        response = sm_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        if "DataCaptureConfig" in response and response["DataCaptureConfig"]["EnableCapture"]:
            logger.info(f"data capture enabled for endpoint config {endpoint_config_name}")

        # Call endpoint to handle
        return invoke_endpoint(endpoint_name)
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", type=str, default=os.environ.get("LOGLEVEL", "INFO").upper())
    parser.add_argument("--import-build-config", type=str, required=True)
    parser.add_argument("--export-test-results", type=str, required=True)
    args, _ = parser.parse_known_args()

    # Configure logging to output the line number and message
    log_format = "%(levelname)s: [%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(format=log_format, level=args.log_level)

    # Load the build config
    with open(args.import_build_config, "r") as f:
        config = json.load(f)

    # Get the endpoint name from sagemaker project name
    endpoint_name = "{}-{}".format(
        config["Parameters"]["SageMakerProjectName"], config["Parameters"]["StageName"]
    )
    results = test_endpoint(endpoint_name)

    # Print results and write to file
    logger.debug(json.dumps(results, indent=4))
    with open(args.export_test_results, "w") as f:
        json.dump(results, f, indent=4)
