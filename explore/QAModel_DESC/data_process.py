import json
import torch

class QA_DataProcessor(object):
    def __init__(self):
        pass

    def retrieve_train_data(self):
        with open("train_inputs.json") as file:
            inputs = json.load(file)
        with open("train_labels.json") as file:
            labels = json.load(file)
        return inputs, labels

    def retrieve_train_data(self, train_file):
        with open(train_file) as json_data:
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
        with open("eval_inputs.json") as file:
            inputs = json.load(file)
        with open("eval_labels.json") as file:
            labels = json.load(file)
        return inputs, labels

    def write_result(self, resFile, result):
        with open(resFile, "w") as json_data:
            json.dump(result, json_data)
        