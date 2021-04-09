"""
ModelHandler defines an example model handler for load and inference requests for MXNet CPU models
"""
from collections import namedtuple
import glob
import json
import logging
import io
import os
import re

import mxnet as mx
import numpy as np
import sys

from pathlib import Path
from gluonts.model.predictor import Predictor
from gluonts.dataset.common import ListDataset

class ModelHandler(object):
    """
    A sample Model handler implementation.
    """

    def __init__(self):
        self.initialized = False
        self.mx_model = None
        self.shapes = None
    
    def load_model(self, model_path):
        try:
            predictor = Predictor.deserialize(Path(model_path))
            print('Model loaded from %s'%model_path)
        except:
            print('Unable to load the model %s'%model_path)
            sys.exit(1)
        return predictor

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self.initialized = True
        properties = context.system_properties
        # Contains the url parameter passed to the load request
        model_dir = properties.get("model_dir") 
        gpu_id = properties.get("gpu_id")

        # Load Gluonts Model
        self.mx_model = self.load_model(model_dir)

    def preprocess(self, request):
        """
        Transform raw input into model input data.
        :param request: list of raw requests
        :return: list of preprocessed model input data
        """
        # Take the input data and pre-process it make it inference ready

        json_list = []
        # for each request
        for idx, data in enumerate(request):
            # Read the bytearray of the jsonline from the input
            jsonline_arr = data.get('body')  
            # Input json is in bytearray, convert it to string
            jsonline_str = jsonline_arr.decode("utf-8")
            # split the json lines
            json_list_request = []
            # for each time series
            for line in io.StringIO(jsonline_str):
                json_record = json.loads(line)
                json_list_request.append(json_record)
            json_list.append(json_list_request)
        return json_list

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data list
        :return: list of inference output in NDArray
        """
        forecast_list = []
        for model_input_request in model_input:
            forecast = list(self.mx_model.predict(ListDataset(
                      model_input_request,
                      freq = self.mx_model.freq
            )))
            forecast_list.append(forecast)
        return forecast_list

    def postprocess(self, inference_output):
        """
        Return predict result in as list.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        ret = []
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # for each request
        for inference_output_request in inference_output:
            ret_request = []
            # for each time series
            for i in inference_output_request:
                l = {}
                l["item_id"] = i.item_id
                l["quantiles"] = {}
                for q in quantiles:
                    l["quantiles"][str(q)] = i.quantile(q).tolist()
                l["mean"] = i.mean.tolist()
                ret_request.append(json.dumps(l))
            ret.append('\n'.join(ret_request) + '\n')
        return ret
        
    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """
        
        model_input = self.preprocess(data)
        model_out = self.inference(model_input)
        return self.postprocess(model_out)

_service = ModelHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
