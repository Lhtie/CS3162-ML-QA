import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLNetTokenizerFast

import data_process

QCModel_file = "QCModel.pth"
Verifier_file = "VerifierModel.pth"
QAModels_file_head = "QAModel_"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class QuestionAnswering(nn.Module):
    def __init__(self, qc_max_len=128, qa_max_len=512, tau=0.01):
        super(QuestionAnswering, self).__init__()
        self.ques_cls = torch.load(QCModel_file, map_location=device)
        self.verifier = torch.load(Verifier_file, map_location=device)
        self.XLNetTokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")
        self.cls_d = 6
        self.cls_type = ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"]
        self.qc_max_len = qc_max_len
        self.qa_max_len = qa_max_len
        self.tau = tau

        self.models = {}
        for token in self.cls_type:
            model_name = QAModels_file_head + token
            sys.path.insert(0, "./" + model_name)
            self.models[token] = torch.load(model_name + '/' + model_name + ".pth", map_location=device)

        self.ques_cls.XLNetBaseModel.eval()
        self.verifier.XLNetBaseModel.eval()
        for token in self.cls_type:
            self.models[token].XLNetBaseModel.eval()

    def forward(self, questions, contexts): # list of batch_size
        batch_size = len(questions)

        tokens = self.XLNetTokenizer(questions, return_tensors="pt", 
                            padding="max_length", truncation=True, max_length=self.qc_max_len)
        inputs = torch.stack([torch.stack([tokens.input_ids[i], tokens.token_type_ids[i], tokens.attention_mask[i]]) for i in range(batch_size)])
        inputs = inputs.to(device)
        cls_outputs = F.softmax(self.ques_cls(inputs), dim=1)
        
        tokens = self.XLNetTokenizer(questions, contexts, return_tensors="pt", 
                            padding="max_length", truncation=True, max_length=self.qa_max_len)
        inputs = torch.stack([torch.stack([tokens.input_ids[i], tokens.token_type_ids[i], tokens.attention_mask[i]]) for i in range(batch_size)])
        inputs = inputs.to(device)
        start_logits, end_logits = torch.zeros(batch_size, self.qa_max_len).to(device), torch.zeros(batch_size, self.qa_max_len).to(device)
        for c, token in enumerate(self.cls_type):
            outputs = self.models[token](inputs)
            start_logits += torch.softmax(outputs.start_logits, dim=1) * cls_outputs[:, c].reshape(-1, 1)
            end_logits += torch.softmax(outputs.end_logits, dim=1) * cls_outputs[:, c].reshape(-1, 1)

        start, end = torch.argmax(start_logits[:, :-1], dim=1), torch.argmax(end_logits[:, :-1], dim=1)
        answers, is_impossible = [], []
        for idx in range(batch_size):
            ans = self.ques_cls.XLNetTokenizer.decode(inputs[idx, 0, start[idx]:end[idx]+1])
            answers.append(ans)

            s_null = start_logits[idx, -1] + end_logits[idx, -1]
            if torch.max(start_logits[idx]) + torch.max(end_logits[idx]) <= s_null + self.tau:
                is_impossible.append(True)
            else: is_impossible.append(False)

        return answers, is_impossible

    def prediction(self, eval_file, res_file, batch_size=48):
        eval_inputs = data_process.eval_DataProcessor(eval_file).retrieve_data()
        print("Eval Data Loaded")

        answers = {}
        batch = 0
        with torch.no_grad():
            for batch_start in range(0, len(eval_inputs), batch_size):
                batch_end = min(len(eval_inputs), batch_start + batch_size)
                questions = [eval_inputs[i]["question"] for i in range(batch_start, batch_end)]
                contexts = [eval_inputs[i]["context"] for i in range(batch_start, batch_end)]
                ans, is_impossible = self(questions, contexts)

                for idx in range(batch_start, batch_end):
                    if is_impossible[idx-batch_start]:
                        answers[eval_inputs[idx]["id"]] = ""
                    else: answers[eval_inputs[idx]["id"]] = ans[idx-batch_start]

                if batch % 50 == 49:
                    print("Prediction {} / {} Finished".format(batch, len(eval_inputs) // batch_size))
                batch = batch + 1
            
        print("Prediction Finished")

        data_process.eval_DataProcessor(eval_file).write_result(res_file, answers)