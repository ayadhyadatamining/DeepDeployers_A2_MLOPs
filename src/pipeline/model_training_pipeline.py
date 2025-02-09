from src.components.data_ingestion import Data_injestion
from src.components.data_processing import Preprocessing
from src.components.model_training import Model_train
import json
import torch
import os
ingestion=Data_injestion()

pre_processing=Preprocessing()

train_model=Model_train()

def Model_train_pipeline(filepath_possitive,filepath_negetive):
    raw_data_positive=ingestion.get_text_from_files(filepath_possitive)

    raw_data_negative=ingestion.get_text_from_files(filepath_negetive)

    processed_data=pre_processing.create_sentiment_dataframe(
                                                                positive_reviews=raw_data_positive,
                                                                negative_reviews=raw_data_negative
                                                             )
    
    model_file_path,best_model_param=train_model.train_model(dataset=processed_data)

    param_file_path=os.path.join(os.getcwd(),"Best_model_param","best_model_params.json")

    with open(param_file_path, "w") as json_file:
        json.dump(best_model_param, json_file, indent=4)

    return model_file_path


def load_best_model_Parameters():
    file_path=os.path.join(os.getcwd(),"Best_model_param","best_model_params.json")
    print("This is the bestparameter json file path")
    with open(file_path, "r") as json_file:
        loaded_params = json.load(json_file)

    return loaded_params
