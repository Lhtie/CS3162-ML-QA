import json
import torch
import QuesCls

class QC_DataProcessor(object):
    def __init__(self):
        self.type_ids = {
            "ABBR": 0, "ENTY": 1, "DESC": 2, "HUM": 3, "LOC": 4, "NUM": 5
        }

    def retrieve_data(self, file_dir):
        with open(file_dir, "r") as file:
            data = file.readlines()
        data = [x.strip() for x in data]

        inputs, labels = [], []
        for raw in data:
            type, qas = raw.split(' ', 1)
            type = self.type_ids[type.split(':', 1)[0]]
            inputs.append(qas)
            labels.append(type)

        return inputs, labels

class Verifier_DataProcessor(object):
    def __init__(self):
        pass

    def retrieve_data(self, file_dir):
        with open(file_dir) as json_data:
            data = json.load(json_data)

        inputs, labels = [], []
        for data in data["data"]:
            for paragraphs in data["paragraphs"]:
                context = paragraphs["context"]
                for qas in paragraphs["qas"]:
                    question = qas["question"]
                    if (qas["is_impossible"]):
                        if len(qas["plausible_answers"]) == 0:
                            answer = data["title"]
                        else: answer = qas["plausible_answers"][0]["text"]
                        inputs.append({"answer": answer, "question": question, "context": context})
                        labels.append(True)
                    else:
                        inputs.append({"answer": qas["answers"][0]["text"], "question": question, "context": context})
                        labels.append(False)

        return inputs, labels

class DataSplitter(object):
    def __init__(self, model_file):
        self.cls_type = ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"]
        self.model = torch.load(model_file)

    def split(self, file_dir, mode):
        with open(file_dir) as json_data:
            data = json.load(json_data)

        inputs, labels = {}, {}
        for token in self.cls_type:
            inputs[token], labels[token] = [], []
        for data in data["data"]:
            for paragraphs in data["paragraphs"]:
                context = paragraphs["context"]
                for qas in paragraphs["qas"]:
                    question =  qas["question"]
                    token = QuesCls.classify(self.model, question)
                    inputs[token].append({"id": qas["id"], "question": question, "context": context})

                    if not qas["is_impossible"]:
                        answer = qas["answers"][0]
                        answer_text = answer["text"]
                        start = answer["answer_start"]
                        end = start + len(answer_text) - 1
                        labels[token].append({"text": answer_text, "start": start, "end": end, "is_impossible": False})
                    else:
                        labels[token].append({"text": "", "start": 0, "end": 0, "is_impossible": True})

            print(data["title"] + " Completed!")

        for token in self.cls_type:
            with open("QAModel_" + token + "/" + mode + "_inputs.json", "w") as file:
                json.dump(inputs[token], file)
            with open("QAModel_" + token + "/" + mode + "_labels.json", "w") as file:
                json.dump(labels[token], file)

class QA_DataProcessor(object):
    def __init__(self):
        pass

    def retrieve_train_data(self, cls_t):
        with open("QAModel_" + cls_t + "/train_inputs.json") as file:
            inputs = json.load(file)
        with open("QAModel_" + cls_t + "/train_labels.json") as file:
            labels = json.load(file)
        return inputs, labels
    
    def retrievee_eval_data(self, cls_t):
        with open("QAModel_" + cls_t + "/eval_inputs.json") as file:
            inputs = json.load(file)
        with open("QAModel_" + cls_t + "/eval_labels.json") as file:
            labels = json.load(file)
        return inputs, labels
    
class eval_DataProcessor(object):
    def __init__(self, eval_file):
        self.eval_file = eval_file

    def retrieve_data(self):
        with open(self.eval_file) as json_data:
            data = json.load(json_data)

        eval_inputs = []
        for data in data["data"]:
            for paragraphs in data["paragraphs"]:
                context = paragraphs["context"]
                for qas in paragraphs["qas"]:
                    question = qas["question"]
                    eval_inputs.append({"id": qas["id"], "question": question, "context": context})

        return eval_inputs

    def write_result(self, resFile, result):
        with open(resFile, "w") as json_data:
            json.dump(result, json_data)