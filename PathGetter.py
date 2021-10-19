import os
from pathlib import Path
path = Path(os.getcwd())
FilePathsfp = os.path.join(path.parent.absolute(), "FilePaths.txt")

def get_fp(dataset_name):
    with open(FilePathsfp, 'r') as paths:
        for path in paths:
            if dataset_name in path:
                return path
    print("Dataset path not found")
    return -1