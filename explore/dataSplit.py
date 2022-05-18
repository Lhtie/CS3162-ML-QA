import data_process

train_data_file = "train-v2.0.json"
eval_data_file = "dev-v2.0.json"
QCModel_file = "QCModel.pth"

util = data_process.DataSplitter(QCModel_file)

util.split(train_data_file, "train")
util.split(eval_data_file, "eval")