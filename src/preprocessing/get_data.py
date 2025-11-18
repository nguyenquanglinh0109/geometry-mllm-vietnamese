import matplotlib.pyplot as plt
import numpy as np
import pickle
from PIL import Image
import os
from tqdm import tqdm
from formalgeo.data import download_dataset

def get_data(data_path: str):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_img(save_img_path: str, data_path: str):
    try:
        data = get_data(data_path)
        for i in tqdm(range(len(data)), desc=f"Saving images into {save_img_path}"):
            img = data[i].get('img', None)
            if img is None:
                img = data[i].get('image', None)
            idx = data[i]['id']
            img = Image.fromarray(img)
            img_path = os.path.join(save_img_path, 'img_' + str(idx) + '.png')
            img.save(img_path)    
    except Exception as e:
        print(f"Error when saving images: {e}")
         
def get_data_unigeo():
    data_path = r"dataset/UniGeo_data/UniGeo"
    for file in os.listdir(data_path):
        true_file = any(x in file for x in ['test', 'val', 'train'])
        if file.endswith('.pk') and true_file:
            save_img_path = os.path.join(data_path, 'image', file[:-3])
            if not os.path.exists(save_img_path):
                os.makedirs(save_img_path)
            save_img(save_img_path, os.path.join(data_path, file))
            
def get_data_formalgeo7k(save_path: str):
    try:
        print("Downloading dataset formalgeo7k_v1...")
        download_dataset(dataset_name="formalgeo7k_v1", datasets_path=save_path)
    except Exception as e:
        print(f"Error when downloading dataset: {e}")

if __name__ == "__main__":
    ## 1. GET DATA UNIGEO
    # get_data_unigeo()
    
    ## 2. GET DATA FORMAL7K
    get_data_formalgeo7k(save_path=r"dataset/formalgeo7k_v1")