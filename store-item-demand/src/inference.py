
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

    data = json.loads(request_body)
    target_test_df = pd.DataFrame(data['value'], index=data['timestamp'])
    target = target_test_df.values
    num_series = target_test_df.shape[1]
    start_dt = target_test_df.index[0]
    test_lst = []
    
    for i in range(0, num_series):
        target_vec = target[:, i]
        dic = {FieldName.TARGET: target_vec, 
               FieldName.START: start_dt} 
        test_lst.append(dic)
        
    test_ds = ListDataset(test_lst, freq='1D')
    
    response_body = {}
    forecast_it = model.predict(test_ds)
    for idx, f in enumerate(forecast_it):
        response_body[f'item_{idx}'] = f.samples.mean(axis=0).tolist()

    return json.dumps(response_body)
