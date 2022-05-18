import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data

from transformers import XLNetModel, XLNetTokenizerFast

import data_process

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class InputType:
    def __init__(self, tokenizer, ctx, max_len):
        tokens = tokenizer(ctx["answer"] + "<sep>" + ctx["question"], ctx["context"], return_tensors="pt", 
                            padding="max_length", truncation=True, max_length=max_len)
        self.input_ids = tokens.input_ids.reshape(-1)
        self.token_type_ids = tokens.token_type_ids.reshape(-1)
        self.attention_mask = tokens.attention_mask.reshape(-1)

        idx = (self.token_type_ids == torch.zeros(1)).nonzero().flatten()
        for i in idx.tolist():
            self.token_type_ids[i] = 4 # qas: 0, ctx: 1, <cls>: 2, <pad>: 3
            if self.input_ids[i] == 4: # id of <sep>: 4
                break

    def convert_to_tensor(self):
        return torch.stack([self.input_ids, self.token_type_ids, self.attention_mask])

class Verifier(nn.Module):
    def __init__(self, max_len=512):
        super(Verifier, self).__init__()

        self.XLNetTokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")
        self.XLNetBaseModel = XLNetModel.from_pretrained("xlnet-base-cased")

        self.config = self.XLNetBaseModel.config
        self.d_model = self.config.d_model
        self.max_len = max_len

        self.classify_layer = nn.Linear(self.d_model, 1)
        self.to(device)

    def forward(self, inputs):
        output = self.XLNetBaseModel(input_ids=inputs[:, 0, :], token_type_ids=inputs[:, 1, :], 
                                    attention_mask=inputs[:, 2, :])
        last_hidden_layer = output.last_hidden_state
        logits = self.classify_layer(last_hidden_layer[:, -1, :])
        return logits.reshape(-1)

    def train(self, train_file, epochs=3, batch_size=48, lr=3e-5, eps=1e-6):
        train_inputs, train_labels = data_process.Verifier_DataProcessor().retrieve_data(train_file)
        inputs = [InputType(self.XLNetTokenizer, _, self.max_len) for _ in train_inputs]
        inputs = torch.stack([ctx.convert_to_tensor() for ctx in inputs])
        train_loader = Data.DataLoader(Data.TensorDataset(inputs, torch.tensor([1.0 if x == True else 0.0 for x in train_labels])),
                        batch_size=batch_size, shuffle=True, num_workers=6)
        print("Train Data Loaded")

        criterion = nn.BCEWithLogitsLoss().to(device)
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

    def retrieve_answer(self, logits, threshold): # input: InputType, output: ModelOutputs
        probs = torch.sigmoid(logits)
        return probs > threshold

    def prediction(self, eval_file, batch_size=48, threshold=0.5):
        eval_inputs, eval_labels = data_process.Verifier_DataProcessor().retrieve_data(eval_file)
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
                    correct += self.retrieve_answer(outputs[idx-batch_start], threshold) == eval_labels[idx]

                if batch % 50 == 49:
                    print("Prediction {} / {} Finished".format(batch, len(inputs) // batch_size))
                batch = batch + 1
            
        print("Prediction Finished, Accuracy: {}".format(correct / len(inputs)))
        