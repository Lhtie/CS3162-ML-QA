import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data

from transformers import XLNetModel, XLNetTokenizerFast

import data_process

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()

    def forward(self, outputs, labels): # outputs: ModelOutputs, labels: Tensor
        s_loss = F.cross_entropy(F.log_softmax(outputs.start_logits, dim=1), labels[:, 0])
        e_loss = F.cross_entropy(F.log_softmax(outputs.end_logits, dim=1), labels[:, 1])
        return (s_loss + e_loss) / 2.0
                
class ModelOutputs:
    def __init__(self, start_logits, end_logits):
        self.start_logits = start_logits
        self.end_logits = end_logits

class InputType:
    def __init__(self, tokenizer, ctx, max_len):
        tokens = tokenizer(ctx["question"], ctx["context"], return_tensors="pt", 
                            padding="max_length", truncation=True, max_length=max_len)
        self.input_ids = tokens.input_ids.reshape(-1)
        self.token_type_ids = tokens.token_type_ids.reshape(-1)
        self.attention_mask = tokens.attention_mask.reshape(-1)

    def convert_to_tensor(self):
        return torch.stack([self.input_ids, self.token_type_ids, self.attention_mask])

class LabelType:
    def __init__(self, tokenizer, ctx, input_ids):
        self.gt_s = torch.zeros(input_ids.shape)
        self.gt_e = torch.zeros(input_ids.shape)
        if ctx["is_impossible"]:
            self.gt_s[-1] = self.gt_e[-1] = 1
        else:
            text_tokens = tokenizer(ctx["text"], return_tensors="pt").input_ids.reshape(-1)
            for idx, token in enumerate(input_ids):
                if token == text_tokens[0]:
                    it = 0
                    while idx + it < len(input_ids) and input_ids[idx+it] == text_tokens[it]:
                        it = it + 1
                    if it == len(text_tokens) - 2: # 2 for <sep> and <cls>
                        self.gt_s[idx] = 1
                        self.gt_e[idx+it-1] = 1
                        break

    def convert_to_tensor(self):
        return torch.stack([self.gt_s, self.gt_e])

class QAModel_ENTY(nn.Module):
    def __init__(self, max_len=512):
        super(QAModel_ENTY, self).__init__()

        self.XLNetTokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")
        self.XLNetBaseModel = XLNetModel.from_pretrained("xlnet-base-cased")

        self.config = self.XLNetBaseModel.config
        self.d_model = self.config.d_model
        self.max_len = max_len
        self.cls_t = "DESC"

        self.start_pred_vec = nn.Linear(self.d_model, 1)
        self.end_pred_vec = nn.Linear(self.d_model, 1)
        self.to(device)

    def forward(self, inputs):
        output = self.XLNetBaseModel(input_ids=inputs[:, 0, :], token_type_ids=inputs[:, 1, :], 
                                    attention_mask=inputs[:, 2, :])
        last_hidden_layer = output.last_hidden_state
        batch_size = last_hidden_layer.shape[0]
        start_logit = self.start_pred_vec(last_hidden_layer).reshape(batch_size, -1)
        end_logit = self.end_pred_vec(last_hidden_layer).reshape(batch_size, -1)
        return ModelOutputs(start_logit, end_logit)

    def train(self, epochs=3, batch_size=48, lr=3e-5, eps=1e-6):
        train_inputs, train_labels = data_process.QA_DataProcessor().retrieve_train_data(self.cls_t)
        inputs = [InputType(self.XLNetTokenizer, _, self.max_len) for _ in train_inputs]
        labels = [LabelType(self.XLNetTokenizer, _, inputs[idx].input_ids) for idx, _ in enumerate(train_labels)]
        inputs = torch.stack([ctx.convert_to_tensor() for ctx in inputs])
        labels = torch.stack([ctx.convert_to_tensor() for ctx in labels])
        train_loader = Data.DataLoader(Data.TensorDataset(inputs, labels),
                        batch_size=batch_size, shuffle=True, num_workers=6)
        print("Train Data Loaded")

        criterion = LossFunction().to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=eps, weight_decay=0)
        self.XLNetBaseModel.train()

        loss_history = []
        for epoch in range(epochs):
            print("Epoch {}".format(epoch))
            running_loss = 0.0
            epoch_loss = []

            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 50 == 49:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f}')
                    epoch_loss.append(running_loss / 50)
                    running_loss = 0.0
            
            loss_history.append(epoch_loss)

        print("Finished Training")
        return loss_history

    def retrieve_answer(self, input, start_logits, end_logits, tau): # input: InputType, output: ModelOutputs
        s_null = start_logits[-1] + end_logits[-1]
        if (torch.max(start_logits) + torch.max(end_logits) <= s_null +  tau):
            return ""
        start, end = torch.argmax(start_logits), torch.argmax(end_logits)
        return self.XLNetTokenizer.decode(input.input_ids[start:end+1])

    def prediction(self, res_file, batch_size=48, tau=1.0):
        eval_inputs, eval_labels = data_process.QA_DataProcessor().retrieve_eval_data(self.cls_t)
        inputs = [InputType(self.XLNetTokenizer, _, self.max_len) for _ in eval_inputs]
        print("Eval Data Loaded")

        self.XLNetBaseModel.eval()
        answers = {}

        batch = 0
        with torch.no_grad():
            for batch_start in range(0, len(inputs), batch_size):
                batch_end = min(len(inputs), batch_start + batch_size)
                batch_inputs = torch.stack([ctx.convert_to_tensor() for ctx in inputs[batch_start:batch_end]])
                batch_inputs = batch_inputs.to(device)
                outputs = self(batch_inputs)

                for idx in range(batch_start, batch_end):
                    answers[eval_inputs[idx]["id"]] = self.retrieve_answer(inputs[idx], 
                            outputs.start_logits[idx-batch_start], outputs.end_logits[idx-batch_start], tau)
                            
                if batch % 50 == 49:
                    print("Prediction {} / {} Finished".format(batch, len(inputs) // batch_size))
                batch = batch + 1
            
        print("Prediction Finished")

        data_process.QA_DataProcessor().write_result(res_file, answers)
        