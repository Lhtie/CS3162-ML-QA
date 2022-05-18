import baseline
import json
import torch

train_data_file = "train-v2.0.json"
eval_data_file = "dev-v2.0.json"
result_file = "prediction.json"
model_file = "baselineModel.pth"

train_mode = False
load_model_from_file = True

if train_mode:
    if load_model_from_file:
        model = torch.load(model_file)
    else: model = baseline.QuestionAnsweringModel()
    loss_history = model.train(train_data_file, batch_size=10)
    torch.save(model, model_file)
    
    with open("baselineModel_loss.json", "w") as file:
        json.dump(loss_history, file)
else:
    model = torch.load(model_file)

model.prediction(eval_data_file, result_file, batch_size=96)

''' tokenizer example
import data_process
train_inputs, train_labels = data_process.DataProcessor(train_data_file).retrieve_train_data()
from transformers import XLNetModel, XLNetTokenizerFast
tokenizer = XLNetTokenizerFast.from_pretrained("xlnet-base-cased")
token = tokenizer.tokenize(train_inputs[0]["context"])
data = tokenizer(train_inputs[0]["context"])
print(tokenizer.convert_ids_to_tokens(data["input_ids"]))


model = baseline.QuestionAnsweringModel(max_len=512)
import data_process
train_inputs, train_labels = data_process.DataProcessor(train_data_file).retrieve_train_data()
inp = baseline.InputType(model.XLNetTokenizer, train_inputs[0], model.max_len)
lab = baseline.LabelType(model.XLNetTokenizer, train_labels[0], inp.input_ids)
answer = model.XLNetTokenizer(train_labels[0]["text"], return_tensors="pt").input_ids.reshape(-1)
print(model.XLNetTokenizer.convert_ids_to_tokens(inp.input_ids))
print(model.XLNetTokenizer.convert_ids_to_tokens(answer))
print(lab.gt_s, lab.gt_e)
'''