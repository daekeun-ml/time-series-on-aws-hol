
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
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions, backtest_metrics
from gluonts.model.predictor import Predictor
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset

def train(args):
    
    # Parse arguments
    epochs = args.epochs
    pred_length = args.pred_length
    num_layers = args.num_layers
    num_cells = args.num_cells
    dropout_rate = args.dropout_rate
    batch_size = args.batch_size
    lr = args.lr
    
    model_dir = args.model_dir
    data_dir = args.data_dir
    num_gpus = args.num_gpus
    output_dir = args.output_dir
    
    device = "gpu" if num_gpus > 0 else "cpu"
    FREQ = 'D'    
    
    target_col = 'Weekly_Sales_sum'
    related_cols = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
       
    # Get training data
    target_train_df = pd.read_csv(os.path.join(data_dir, 'target_train.csv'), index_col=0, header=[0,1])
    related_train_df = pd.read_csv(os.path.join(data_dir, 'related_train.csv'), index_col=0, header=[0,1])
    store_df = pd.read_csv(os.path.join(data_dir, 'item.csv'), index_col=0)
    
    num_steps, num_series = target_train_df.shape
    target = target_train_df.values

    start_train_dt = target_train_df.index[0]
    custom_ds_metadata = {'num_series': num_series, 
                          'num_steps': num_steps,
                          'prediction_length': pred_length,
                          'freq': FREQ,
                          'start': [start_train_dt for _ in range(num_series)] 
                         }

    # Prepare GlounTS Dataset
    related_list = [related_train_df[c].values for c in related_cols]

    train_lst = []
    for i in range(0, num_series):
        target_vec = target[:-pred_length, i]
        related_vecs = [related[:-pred_length, i] for related in related_list]
        item = store_df.loc[i+1]
        dic = {FieldName.TARGET: target_vec, 
               FieldName.START: start_train_dt,
               FieldName.FEAT_DYNAMIC_REAL: related_vecs,
               FieldName.FEAT_STATIC_CAT: [item[0]],
               FieldName.FEAT_STATIC_REAL: [item[1]]
              } 
        train_lst.append(dic)

    test_lst = []
    for i in range(0, num_series):
        target_vec = target[:, i]
        related_vecs = [related[:, i] for related in related_list]
        item = store_df.loc[i+1]    
        dic = {FieldName.TARGET: target_vec, 
               FieldName.START: start_train_dt,
               FieldName.FEAT_DYNAMIC_REAL: related_vecs,
               FieldName.FEAT_STATIC_CAT: [item[0]],
               FieldName.FEAT_STATIC_REAL: [item[1]]
              } 
        test_lst.append(dic)

    train_ds = ListDataset(train_lst, freq=FREQ)
    test_ds = ListDataset(test_lst, freq=FREQ)   

    # Define Estimator    
    trainer = Trainer(
        ctx=device,
        epochs=epochs,
        learning_rate=lr,
        batch_size=batch_size
    )
    
    deepar_estimator = DeepAREstimator(freq=FREQ, 
                                       prediction_length=pred_length,
                                       use_feat_dynamic_real=True,
                                       use_feat_static_cat=True,
                                       use_feat_static_real=True,
                                       cardinality=[3],
                                       num_cells=30,
                                       distr_output=StudentTOutput(),
                                       trainer=trainer)
    # Train the model
    deepar_predictor = deepar_estimator.train(train_ds)
    
    # Evaluate trained model on test data
    forecast_it, ts_it = make_evaluation_predictions(test_ds, deepar_predictor, num_samples=100)
    forecasts = list(forecast_it)
    tss = list(ts_it)
    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))

    metrics = ['RMSE', 'MAPE', 'wQuantileLoss[0.1]', 'wQuantileLoss[0.5]', 'wQuantileLoss[0.9]', 'mean_wQuantileLoss']
    metrics_dic = dict((key,value) for key, value in agg_metrics.items() if key in metrics)
    print(json.dumps(metrics_dic, indent=2))

    # Save the model
    deepar_predictor.serialize(pathlib.Path(model_dir))
    return deepar_predictor


def parse_args():
    parser = argparse.ArgumentParser()

    # Hyperparameter Setting
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--pred_length', type=int, default=12)    
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_cells', type=int, default=30)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--batch_size', type=float, default=32)
    parser.add_argument('--lr', type=float, default=0.001) 
    
    # SageMaker Container Environment
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num_gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    
    args = parser.parse_args()
    return args    

if __name__ == '__main__':
    args = parse_args()
    train(args)    
