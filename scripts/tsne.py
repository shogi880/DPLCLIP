from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

import clip
import glob
import random
from PIL import Image
import torch
from tqdm import tqdm 
import numpy as np
from PIL import ImageFile
import cloudpickle
ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = ['PACS', 'VLCS', 'terra_incognita', 'office_home']
MODELS = ['clip']
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_model_from_checkpoint(file_name):
    with open(file_name, 'rb') as f:
        model = cloudpickle.load(f)
    return model

def get_model(model_name):
    if model_name == 'clip':
        model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

ckpt_dir="images"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

def select_image(select_num=100, dataset='PACS'):
    dataset_folders = glob.glob(f'~/dataset/{dataset}/*')
    seleckted_image_path = []
    labels = []
    for domain in dataset_folders:
        classes = glob.glob(domain+'/*')
        for i, class_name in enumerate(classes):
            # labels.append(class_name.split('/')[-1])
            imgs = glob.glob(class_name+'/*')
            if len(imgs) < select_num:
                seleckted_image_path += imgs
                labels += [i] * len(imgs)
            else:
                seleckted_image_path += random.sample(imgs, select_num)
                labels += [i] * select_num
    return seleckted_image_path, np.array(labels)

def get_image_feature(image_paths, model, preprocess):
    image_features = []
    imgs = []
    for i in tqdm(range(0, len(image_paths), 128)):
        images = []
        if i + 128 > len(image_paths):
            for image_path in image_paths[i:]:
                images.append(preprocess(Image.open(image_path)).to(device))    
        else:
            for image_path in image_paths[i:i+128]:
                images.append(preprocess(Image.open(image_path)).to(device))
        imgs.append(model.encode_image(torch.stack(images)).to('cpu').detach().numpy().copy())
    import ipdb; ipdb.set_trace()
    img = np.concatenate(imgs)
    return img

def tsne(image_features, labels, name, model_name):
    X_tsne = TSNE(n_components=2,random_state=33).fit_transform(image_features)
    X_pca = PCA(n_components=2).fit_transform(image_features)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=labels, label="t-SNE", cmap='jet', s=15, alpha=0.5)
    plt.legend()
    plt.subplot(122)
    plt.scatter(X_pca[:, 0], X_pca[:, 1],c=labels, label="PCA")
    plt.legend()
    plt.colorbar()
    plt.savefig(f'images/{name}_{model_name}.png', dpi=120)

for model_name in MODELS:
    for dataset in DATASETS:
        image_paths, labels = select_image(select_num=100, dataset=dataset)
        model, preprocess = get_model(model_name)
        image_features = get_image_feature(image_paths, model, preprocess)
        tsne(image_features, labels, dataset, model_name)