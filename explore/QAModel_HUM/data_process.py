import json
import torch

class QA_DataProcessor(object):
    def __init__(self):
        pass

    def retrieve_train_data(self, cls_t):
        with open("train_inputs.json") as file:
            inputs = json.load(file)
        with open("train_labels.json") as file:
            labels = json.load(file)
        return inputs, labels
    
    def retrieve_eval_data(self, cls_t):
        with open("eval_inputs.json") as file:
            inputs = json.load(file)
        with open("eval_labels.json") as file:
            labels = json.load(file)
        return inputs, labels

    def write_result(self, resFile, result):
        with open(resFile, "w") as json_data:
            json.dump(result, json_data)
        