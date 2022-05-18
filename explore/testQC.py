import QuesCls
import json
import torch

train_data_file = "Question_Classification_Data/Training_set.txt"
eval_data_file = "Question_Classification_Data/Test_set.txt"
model_file = "QCModel.pth"

train_mode = False
load_model_from_file = True

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

if train_mode:
    if load_model_from_file:
        model = torch.load(model_file, map_location=device)
    else: model = QuesCls.QuestionClassificator()
    loss_history = model.train(train_data_file, batch_size=60)
    torch.save(model, model_file)
    
    with open("QCModel_loss.json", "w") as file:
        json.dump(loss_history, file)
else:
    model = torch.load(model_file, map_location=device)

model.prediction(eval_data_file, batch_size=96)