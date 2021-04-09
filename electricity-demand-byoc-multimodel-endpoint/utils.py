# Import libraries
import boto3
import json
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import os
import pandas as pd
import random
import sys
import tarfile
import shutil
import zipfile

from botocore.client import ClientError
from gluonts.model.trivial.mean import MeanPredictor
from gluonts.model.seasonal_naive import SeasonalNaivePredictor
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.prophet import ProphetPredictor
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from gluonts.dataset.common import ListDataset
from itertools import islice
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
from pathlib import Path
from urllib.request import urlretrieve
from gluonts.mx.distribution import StudentTOutput, PoissonOutput, NegativeBinomialOutput

# fix some plot issues caused by Prophet model
# pls refer to https://darektidwell.com/typeerror-float-argument-must-be-a-string-or-a-number-not-period-facebook-prophet-and-pandas/
pd.plotting.register_matplotlib_converters()

# set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
mx.random.seed(42)

# Download Data
DATA_HOST = "https://archive.ics.uci.edu"
DATA_PATH = "/ml/machine-learning-databases/00321/"
ARCHIVE_NAME = "LD2011_2014.txt.zip"
ARCHIVE_PATH = './data/' + ARCHIVE_NAME
FILE_NAME = ARCHIVE_NAME[:-4]
FILE_PATH = './data/' + FILE_NAME
FILE_DIR = os.path.dirname(FILE_PATH)

os.makedirs(FILE_DIR, exist_ok=True)

def progress_report_hook(count, block_size, total_size):
    mb = int(count * block_size // 1e6)
    if count % 500 == 0:
        sys.stdout.write("\r{} MB downloaded".format(mb))
        sys.stdout.flush()

if not os.path.isfile(FILE_PATH):
    print("downloading dataset (258MB), can take a few minutes depending on your connection")
    urlretrieve(DATA_HOST + DATA_PATH + ARCHIVE_NAME, ARCHIVE_PATH, reporthook=progress_report_hook)

    print("\nextracting data archive")
    zip_ref = zipfile.ZipFile(ARCHIVE_PATH, 'r')
    zip_ref.extractall(FILE_DIR + '/')
    zip_ref.close()
else:
    print("File found skipping download")
    

def split_train_test_data(timeseries,
                          start_training,
                          end_training,
                          num_test_windows):
    # create training data.
    training_data = [
    {
        "start": str(start_training),
        "target": ts[start_training:end_training].tolist(),
        "feat_static_cat": [id],
        "item_id": str(id)
    }
    for id, ts in enumerate(timeseries)
    ]
    
    # create testing data.
    test_data = [
        {
            "start": str(start_training),
            "target": ts[start_training:end_training + k * prediction_length*ts.index.freq].tolist(),
            "feat_static_cat": [id],
            "item_id": str(id)
        }
        for k in range(1, num_test_windows + 1)
        for id, ts in enumerate(timeseries)
    ]
    
    return training_data, test_data

# we use 2 hour frequency for the time series
freq = '2H'

# we predict for 1 day
prediction_length = 1 * 12

# we also use 7 days as context length, this is the number of state updates accomplished before making predictions
context_length = 7 * 12

# The moving window for forecast
num_test_windows = 1

# training/test Split
start_training = pd.Timestamp("2014-01-01 00:00:00", freq=freq)
end_training = pd.Timestamp("2014-01-31 00:00:00", freq=freq)

# number of time series selected
n_timeseries = 100

account_id = boto3.client('sts').get_caller_identity()['Account']
region = boto3.Session().region_name
bucket = 'sagemaker-{}-{}'.format(region, account_id)
prefix = 'demo-multimodel-gluonts-endpoint'
output_dir = 'data'
models_dir = "models"

def write_dicts_to_file(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as fp:
        for d in data:
            fp.write(json.dumps(d).encode("utf-8"))
            fp.write("\n".encode('utf-8'))

def upload_to_s3(bucket, prefix, path):
    s3_client = boto3.client('s3')
    s3_client.upload_file(Filename=path, Bucket=bucket, Key=os.path.join(prefix, path))


def train_and_test(training_data, 
                   test_data,
                   freq,
                   num_test_windows,
                   model,
                   require_train=False
                   ):
    forecasts = []
    tss = []
    training_data = ListDataset(
                  training_data,
                  freq = freq
        )
    test_data = ListDataset(
        test_data,
        freq = freq
    )
    if require_train:
        predictor = model.train(training_data=training_data)
    else:
        predictor = model
    
    # Save the model locally for later deployment.
    model_name = model.__class__.__name__
    model_path = Path(f"models/{model_name}")
    os.makedirs(model_path, exist_ok=True)
    predictor.serialize(model_path)
    
    # Do the forecast on the test set.
    forecast_it, ts_it = make_evaluation_predictions(test_data, predictor=predictor, num_samples=100)
    forecasts.extend(list(forecast_it))
    tss.extend(list(ts_it))
        
    return forecasts, tss


def plot_forecasts(tss, forecasts, past_length, start, stop, step, title):
    for target, forecast in islice(zip(tss, forecasts), start, stop, step):
        ax = target[-past_length:].plot(figsize=(12, 5), linewidth=2)
        forecast.plot(color='g')
        plt.title(title)
        plt.grid(which='both')
        plt.legend(["observations", "median prediction", "90% confidence interval", "50% confidence interval"])
        plt.show()


s3 = boto3.resource('s3')
try:
    s3.meta.client.head_bucket(Bucket=bucket)
except ClientError:
    s3.create_bucket(Bucket=bucket,
                     CreateBucketConfiguration={
                         'LocationConstraint': region
                     })
