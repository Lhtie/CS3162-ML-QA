import json
import torch
from torch.utils.data import Dataset

class DataProcessor(object):
    def __init__(self, trainFile=None, evalFile=None, resFile=None):
        self.trainFile = trainFile
        self.evalFile = evalFile
        self.resFile = resFile

    def retrieve_train_data(self):
        with open(self.trainFile) as json_data:
            data = json.load(json_data)

        train_inputs, train_labels = [], []
        for data in data["data"]:
            for paragraphs in data["paragraphs"]:
                context = paragraphs["context"]
                for qas in paragraphs["qas"]:
                    question = qas["question"]
                    train_inputs.append({"question": question, "context": context})
                    if (qas["is_impossible"]):
                        train_labels.append({"text": 0, "start": 0, "end": 0, "is_impossible": True})
                    else:
                        answer = qas["answers"][0]
                        answer_text = answer["text"]
                        start = answer["answer_start"]
                        end = start + len(answer_text) - 1
                        train_labels.append({"text": answer_text, "start": start, "end": end, "is_impossible": False})

        return train_inputs, train_labels

    def retrieve_eval_data(self):
        with open(self.evalFile) as json_data:
            data = json.load(json_data)

        eval_inputs = []
        for data in data["data"]:
            for paragraphs in data["paragraphs"]:
                context = paragraphs["context"]
                for qas in paragraphs["qas"]:
                    question = qas["question"]
                    eval_inputs.append({"id": qas["id"], "question": question, "context": context})

        return eval_inputs

    def write_result(self, result):
        with open(self.resFile, "w") as json_data:
            json.dump(result, json_data)

class DataSet(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        assert len(self.inputs) == len(self.labels)
        return len(self.inputs)

    def __getitem__(self, idx):
        return (self.inputs[idx], self.labels[idx])
