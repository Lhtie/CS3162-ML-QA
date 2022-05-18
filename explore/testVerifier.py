import Verifier
import json
import torch

train_data_file = "../train-v2.0.json"
eval_data_file = "../dev-v2.0.json"
model_file = "VerifierModel.pth"

train_mode = True
load_model_from_file = True

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

if train_mode:
    if load_model_from_file:
        model = torch.load(model_file, map_location=device)
    else: model = Verifier.Verifier()
    loss_history = model.train(train_data_file, batch_size=12)
    torch.save(model, model_file)
    
    with open("VerifierModel_loss.json", "w") as file:
        json.dump(loss_history, file)
else:
    model = torch.load(model_file, map_location=device)

model.prediction(eval_data_file, batch_size=96)