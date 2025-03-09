import json 

def load_jsonl(file_path):
    data = {}
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.update(json.loads(line.strip()))
    return data