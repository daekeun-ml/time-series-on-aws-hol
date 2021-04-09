import io
import json
from typing import List

import pytest
from gluonts.model.predictor import Predictor

from model_handler import ModelHandler

# Define some helper classes
class Context(object):
    def __init__(self, system_properties):
        self.system_properties = system_properties

class Request(object):
    def __init__(self, body):
        self.body = body

    # get method 
    def get(self, para="body"): 
        return self.body 

@pytest.fixture
def model_hdlr():
    model_hdlr = ModelHandler()
    return model_hdlr

@pytest.fixture
def context():
    context=Context({"model_dir":"../models/MeanPredictor",
                 "gpu_id": 1})
    return context

@pytest.fixture
def request_body():
    return b"""{"start": "2014-01-01 00:00:00", "target": [2.6967005076142154, 2.8553299492385804, 2.53807106598985, 3.0139593908629454, 0.7931472081218263, 0.47588832487309496, 0.47588832487309496, 0.47588832487309496, 2.2208121827411174, 2.8553299492385804, 2.53807106598985, 2.8553299492385804], "feat_static_cat": [0], "item_id": "0"}
    {"start": "2014-01-01 00:00:00", "target": [23.648648648648624, 23.204125177809388, 21.8705547652916, 21.78165007112375, 24.00426742532005, 29.960881934566153, 31.916785206258883, 28.982930298719776, 26.76031294452345, 26.6714082503556, 27.91607396870555, 25.604551920341397], "feat_static_cat": [1], "item_id": "1"}"""

@pytest.fixture
def rqst(request_body: bytes):
    rqst =Request(request_body)
    return rqst

def test_load_model(model_hdlr: ModelHandler):
    predictor = model_hdlr.load_model("../models/DeepAREstimator")
    assert isinstance(predictor, Predictor)

def test_initialize(model_hdlr: ModelHandler,
                    context: Context):
    model_hdlr.initialize(context)
    predictor = model_hdlr.mx_model
    assert isinstance(predictor, Predictor)

def test_preprocess(model_hdlr: ModelHandler,
                   rqst: Request):
    results = model_hdlr.preprocess([rqst])
    assert len(results) == 1

@pytest.mark.parametrize("num_samples,quantiles", [(5, ["0.4", "0.6"]), (25, ["0.2", "0.5", "0.8"])])
def test_handle(model_hdlr: ModelHandler,
                   rqst: Request,
                   context: Context,
                   num_samples: int,
                   quantiles: List[str]):
    model_hdlr.initialize(context)
    results_list = model_hdlr.handle([rqst], context)
    
    # Make sure each JSON line is somewhat correct.
    # for each request
    for results_list_request in results_list:
        for line in io.StringIO(results_list_request):
            d = json.loads(line)
            for quantile in quantiles:
                assert quantile in d["quantiles"]
                assert len(d["quantiles"][quantile]) == model_hdlr.mx_model.prediction_length
            assert ("mean" in d) and (len(d["mean"]) == model_hdlr.mx_model.prediction_length)

    # Print results; need to run pytest -v -rA --tb=short ...
    print(results_list)
    return results_list
