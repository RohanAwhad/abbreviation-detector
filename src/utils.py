import json

def load_json(filepath):
    with open(filepath, 'r') as fs:
        return json.load(fs)