import pickle
import json

def writeCachedDict(file_path, dict):
    out_file = open(file_path, 'wb')
    pickle.dump(dict, out_file)
    out_file.close()

def readCachedDict(file_path):
    f = open(file_path, 'rb')
    dict = pickle.load(f)
    return dict

# note this overwrites the previous cached dict
def writeReadableCachedDict(file_path, dict):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dict, f, ensure_ascii=False, indent=4)

def readReadableCachedDict(file_path):
    with open(file_path) as f:
        dict = json.load(f)
    return dict
