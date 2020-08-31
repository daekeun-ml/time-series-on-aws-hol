
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
from gluonts.distribution import DistributionOutput, StudentTOutput, NegativeBinomialOutput, GaussianOutput
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
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

    related_cols = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment'] 
    item_cols = ['Type', 'Size'] 
    FREQ = 'W'
    pred_length = 12

    data = json.loads(request_body)    
    
    target_test_df = pd.DataFrame(data['target_values'], index=data['timestamp'])
    related_test_df = pd.DataFrame(data['related_values'], index=data['timestamp'])
    item_df = pd.DataFrame(data['item'], index=data['store_id'])
    item_df.columns = item_cols
        
    target = target_test_df.values
    num_steps, num_series = target_test_df.shape
    start_dt = target_test_df.index[0]
    
    num_related_cols = len(related_cols)
    num_features_per_feature = int(related_test_df.shape[1] / num_related_cols)
    related_list = []

    for feature_idx in range(0, num_related_cols):
        start_idx = feature_idx * num_features_per_feature
        end_idx = start_idx + num_features_per_feature
        related_list.append(related_test_df.iloc[:, start_idx:end_idx].values)

    test_lst = []
    for i in range(0, num_series):
        target_vec = target[:-pred_length, i]
        related_vecs = [related[:, i] for related in related_list]
        item = item_df.loc[i+1]    
        dic = {FieldName.TARGET: target_vec, 
               FieldName.START: start_dt,
               FieldName.FEAT_DYNAMIC_REAL: related_vecs,
               FieldName.FEAT_STATIC_CAT: [item[0]],
               FieldName.FEAT_STATIC_REAL: [item[1]]
              } 
        test_lst.append(dic)

    test_ds = ListDataset(test_lst, freq=FREQ)

    response_body = {}
    forecast_it = model.predict(test_ds)

    for idx, f in enumerate(forecast_it):
        response_body[f'store_{idx}'] = f.samples.mean(axis=0).tolist()

    return json.dumps(response_body)
