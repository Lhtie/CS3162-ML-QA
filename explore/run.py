import modeling

eval_file = "dev-v2.0.json"
res_file = "prediction.json"

model = modeling.QuestionAnswering()
model.prediction(eval_file, res_file, batch_size=96)