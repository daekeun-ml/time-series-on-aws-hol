
import os
import pandas as pd
import gluonts 
import numpy as np
import argparse
import json
import pathlib
from mxnet import gpu, cpu
from mxnet.context import num_gpus
import matplotlib.pyplot as plt

from gluonts.dataset.util import to_pandas
from gluonts.mx.distribution import DistributionOutput, StudentTOutput, NegativeBinomialOutput, GaussianOutput
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions, backtest_metrics
from gluonts.model.predictor import Predictor
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset


def model_fn(model_dir):
    path = pathlib.Path(model_dir)   
    predictor = Predictor.deserialize(path)
    print("model was loaded successfully")
    return predictor


def transform_fn(model, request_body, content_type='application/json', accept_type='application/json'):
    
    related_cols = ['holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'weather_main', 'weather_description']    
    FREQ = 'H'
    pred_length = 24*7

    data = json.loads(request_body)

    target_test_df = pd.DataFrame(data['target_values'], index=data['timestamp'])
    related_test_df = pd.DataFrame(data['related_values'], index=data['timestamp'])
    related_test_df.columns = related_cols
        
    target = target_test_df.values
    related = related_test_df.values
    num_series = target_test_df.shape[1]
    start_dt = target_test_df.index[0]

    related_list = [related_test_df[c].values for c in related_cols]    
    test_lst = []

    target_vec = target.squeeze()
    related_vecs = [related.squeeze() for related in related_list]
    dic = {FieldName.TARGET: target_vec, 
           FieldName.START: start_dt,
           FieldName.FEAT_DYNAMIC_REAL: related_vecs
          } 
    test_lst.append(dic)
    test_ds = ListDataset(test_lst, freq=FREQ)
    
    response_body = {}
    forecast_it = model.predict(test_ds)
    forecast = list(forecast_it)
    response_body['out'] = forecast[0].samples.mean(axis=0).tolist()
    return json.dumps(response_body)
