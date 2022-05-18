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
        return F.cross_entropy(F.log_softmax(outputs, dim = 1), labels)

class InputType:
    def __init__(self, tokenizer, ctx, max_len):
        tokens = tokenizer(ctx, return_tensors="pt", 
                            padding="max_length", truncation=True, max_length=max_len)
        self.input_ids = tokens.input_ids.reshape(-1)
        self.token_type_ids = tokens.token_type_ids.reshape(-1)
        self.attention_mask = tokens.attention_mask.reshape(-1)

    def convert_to_tensor(self):
        return torch.stack([self.input_ids, self.token_type_ids, self.attention_mask])

class LabelType:
    def __init__(self, ctx, cls_d):
        self.gt = torch.zeros(cls_d)
        self.gt[ctx] = 1

    def convert_to_tensor(self):
        return self.gt

class QuestionClassificator(nn.Module):
    def __init__(self, max_len=128):
        super(QuestionClassificator, self).__init__()

        self.XLNetTokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")
        self.XLNetBaseModel = XLNetModel.from_pretrained("xlnet-base-cased")

        self.config = self.XLNetBaseModel.config
        self.d_model = self.config.d_model
        self.max_len = max_len
        self.cls_d = 6
        self.cls_type = ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"]

        self.classify_layer = nn.Linear(self.d_model, self.cls_d)
        self.to(device)

    def forward(self, inputs):
        output = self.XLNetBaseModel(input_ids=inputs[:, 0, :], token_type_ids=inputs[:, 1, :], 
                                    attention_mask=inputs[:, 2, :])
        last_hidden_layer = output.last_hidden_state
        batch_size = last_hidden_layer.shape[0]
        logits = self.classify_layer(last_hidden_layer)
        return logits[:, -1, :].reshape(batch_size, -1)

    def train(self, train_file, epochs=3, batch_size=48, lr=3e-5, eps=1e-6):
        train_inputs, train_labels = data_process.QC_DataProcessor().retrieve_data(train_file)
        inputs = [InputType(self.XLNetTokenizer, _, self.max_len) for _ in train_inputs]
        labels = [LabelType(_, self.cls_d) for _ in train_labels]
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
                if i % 10 == 9:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                    epoch_loss.append(running_loss / 10)
                    running_loss = 0.0
            
            loss_history.append(epoch_loss)

        print("Finished Training")
        return loss_history

    def retrieve_answer(self, logits): # input: InputType, output: ModelOutputs
        probs = F.softmax(logits, dim = 0)
        return torch.argmax(probs)

    def prediction(self, eval_file, batch_size=48):
        eval_inputs, eval_labels = data_process.QC_DataProcessor().retrieve_data(eval_file)
        inputs = [InputType(self.XLNetTokenizer, _, self.max_len) for _ in eval_inputs]
        print("Eval Data Loaded")

        self.XLNetBaseModel.eval()
        batch = 0
        correct = 0.0
        with torch.no_grad():
            for batch_start in range(0, len(inputs), batch_size):
                batch_end = min(len(inputs), batch_start + batch_size)
                batch_inputs = torch.stack([ctx.convert_to_tensor() for ctx in inputs[batch_start:batch_end]])
                batch_inputs = batch_inputs.to(device)
                outputs = self(batch_inputs)

                for idx in range(batch_start, batch_end):
                    res = self.retrieve_answer(outputs[idx-batch_start])
                    correct += res == eval_labels[idx]

                if batch % 50 == 49:
                    print("Prediction {} / {} Finished".format(batch, len(inputs) // batch_size))
                batch = batch + 1
            
        print("Prediction Finished, Accuracy: {}".format(correct / len(inputs)))

def classify(model, qas):
    model.XLNetBaseModel.eval()
    inputs = torch.stack([InputType(model.XLNetTokenizer, qas, model.max_len).convert_to_tensor()])
    inputs = inputs.to(device)
    outputs = model(inputs)
    return model.cls_type[model.retrieve_answer(outputs[0])]