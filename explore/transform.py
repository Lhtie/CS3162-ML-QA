import json

cls_type = ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"]

for token in cls_type:
    with open("QAModel_" + token + "/eval_inputs.json") as file:
        inputs = json.load(file)
    lookup = {}
    for ctx in inputs:
        lookup[ctx["id"]] = True
    with open("dev-v2.0.json") as file:
        data = json.load(file)

    out = {}
    out["version"] = "v2.0"
    out["data"] = []
    for data in data["data"]:
        out_data = {}
        out_data["title"] = data["title"]
        out_paragraphs = []
        for paragraphs in data["paragraphs"]:
            out_para = {}
            out_qas = []
            for qas in paragraphs["qas"]:
                if not lookup.get(qas["id"]) == None:
                    out_qas.append(qas)
            out_para["qas"] = out_qas
            out_para["context"] = paragraphs["context"]
            out_paragraphs.append(out_para)
        out_data["paragraphs"] = out_paragraphs
        out["data"].append(out_data)
    
    with open("QAModel_" + token + "/dev-specified.json", "w") as file:
        json.dump(out, file)