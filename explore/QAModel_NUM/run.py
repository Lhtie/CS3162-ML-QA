import QAModel_NUM
import json
import torch

result_file = "prediction.json"
model_file = "QAModel_NUM.pth"

train_mode = True
load_model_from_file = True

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

if train_mode:
    if load_model_from_file:
        model = torch.load(model_file, map_location=device)
    else: model = QAModel_NUM.QAModel_NUM()
    loss_history = model.train(epochs=3, batch_size=12)
    torch.save(model, model_file)
    
    with open("QAModel_NUM_loss.json", "w") as file:
        json.dump(loss_history, file)
else:
    model = torch.load(model_file, map_location=device)

model.prediction(result_file, batch_size=48)
