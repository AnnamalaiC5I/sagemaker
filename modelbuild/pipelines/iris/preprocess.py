"""Feature engineers the abalone dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile

import boto3
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


# Since we get a headerless CSV file we specify the column names here.
feature_columns_names = [
    "sepal_length", 
    "sepal_width", 
    "petal_length", 
    "petal_width",   
]

label_column = "species"

# feature_columns_dtype = {
#      "sepal_length": dtype.float64, 
#     "sepal_width": dtype.float64, 
#     "petal_length": dtype.float64, 
#      "petal_width": dtype.float64,  
# }
# label_column_dtype = {"species": str}


def merge_two_dicts(x, y):
    """Merges two dicts, returning a new copy."""
    z = x.copy()
    z.update(y)
    return z


if __name__ == "__main__":
    logger.debug("Starting preprocessing.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/IRIS.csv"
    #custom code 7
    s3 = boto3.resource("s3",aws_access_key_id='AKIAUJKJ5ZIQJK6XUY6X', 
                      aws_secret_access_key='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', 
                      region_name='ap-south-1')
    #s3_client = boto3.client('s3')
    #obj = s3_client.get_object(Bucket=bucket,Key='dataset/IRIS.csv')
    s3.Bucket(bucket).download_file(key, fn) 

    logger.debug("Reading downloaded data.")
    df = pd.read_csv(
        fn,
        #header=None,
        #names=feature_columns_names + [label_column],
        #dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype),
    )
    os.unlink(fn)

    logger.debug("Defining transformers.")
    numeric_features = list(feature_columns_names)
    #numeric_features.remove("species")
    # numeric_transformer = Pipeline(
    #     steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    # )

    categorical_features = ["species"]
    # categorical_transformer = Pipeline(
    #     steps=[
    #         ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    #         ("onehot", OneHotEncoder(handle_unknown="ignore")),
    #     ]
    # )
    categoricsl_transformer = LabelEncoder()

   
    
    logger.info("Applying transforms on Species")
    df[label_column] = categoricsl_transformer.fit_transform(df[label_column])
    y = df.pop(label_column)
    X_pre = df.copy()
    y_pre = y.to_numpy().reshape(len(y), 1)

    X = np.concatenate((y_pre, X_pre), axis=1)

    logger.info("Splitting %d rows of data into train, validation, test datasets.", len(X))
    np.random.shuffle(X)
    train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])

    logger.info("Writing out datasets to %s.", base_dir)
    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(
        f"{base_dir}/validation/validation.csv", header=False, index=False
    )
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
