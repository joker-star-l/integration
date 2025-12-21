import logging
import pandas as pd
from joblib import dump, load
from pickle import UnpicklingError, PickleError


def load_model(file):
    try:
        model = load(file)
        return model
    except UnpicklingError as e:
        logging.error(e)
        return None


def save_model(model, name):
    try:
        dump(model, name)
    except PickleError as e:
        logging.error(e)
        return None
    

datasets = {}
def load_dataset(data_path):
    if data_path not in datasets:
        datasets[data_path] = pd.read_csv(data_path)
    return datasets[data_path]
