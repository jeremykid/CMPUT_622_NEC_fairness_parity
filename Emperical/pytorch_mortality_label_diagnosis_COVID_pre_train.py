from math import sqrt, ceil
from collections import defaultdict
import os, sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score
from tqdm import tqdm, trange

from torch.optim import Adam

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)


from typing import Type, Any, Callable, Union, List, Optional

from Torch_auto_ecg_model import ECG_ResNet
from helper import pytorch_data_loader

from ecg_survival_loader_agsx import ECG_Dataset_Diagonal_labels
# parameter
GPU_NUM = '5'
LABEL_NUM = 1

device = torch.device("cuda:"+GPU_NUM if torch.cuda.is_available() else "cpu")

sys.path.append(os.path.abspath("../../src/features"))
from ecg_splits import extract_data_splits

covid_ecg_df = pd.read_pickle('/data/padmalab/ecg/data/raw/new_2022_data/Population_data/rmt22884_covid_ecg_ISD_df.pickle')
data_train, data_val, data_test = extract_data_splits(covid_ecg_df)

data_train_label = data_train.loc[data_train_label.index.tolist(), 'COVID_label']
data_val_label = data_val.loc[data_val_label.index.tolist(), 'COVID_label']
data_train = data_train.loc[data_train_label.index.tolist()]
data_val = data_val.loc[data_val_label.index.tolist()]

np_ECG_path = '/data/padmalab/ecg/data/raw/new_2022_data/ecg_np/%s.xml.npy.gz'

train_loader = DataLoader(ECG_Dataset_Diagonal_labels(
    data_train, 
    age_sex_path_df=covid_ecg_df[['AGE', 'SEX']], 
    y=data_train_label, np_path=np_ECG_path),
    batch_size=512, shuffle=True, num_workers = 3)
val_loader = DataLoader(ECG_Dataset_Diagonal_labels(
    data_val, 
    age_sex_path_df=covid_ecg_df[['AGE', 'SEX']], 
    y=data_val_label, np_path=np_ECG_path),
    batch_size=512, shuffle=True, num_workers = 3)

def validate(device, model, val_loader):
    model.eval()
    val_loss = 0
    lossfun = nn.BCELoss()
    with torch.no_grad():
        with tqdm(total=len(val_loader)) as pbar:
            for Id, xi, yi in val_loader:
                xi[0] = xi[0].to(device)
                xi[1] = xi[1].to(device)
                yi = yi.to(torch.float32)
                yi = yi.to(device)
                y_pred = model({0:xi[0], 1:xi[1]})
                loss = lossfun(y_pred, yi)
                val_loss += loss.item()
                pbar.set_postfix_str(f"loss = {loss.item():.4f}")
                pbar.update(1)
                
    return val_loss

from Torch_auto_ecg_model import ECG_ResNet, initialize_weights

model = torch.load('./model/torch_diagnosis_pre_covid.model')

############ frozen ############
for name, param in model.named_parameters():
    if name not in ['dense_agsx.weight', 'dense_agsx.bias', 'dense.weight', 'dense.bias']:
        param.requires_grad = False
########### end of frozen ############

model = model.to(device)

lossfun = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=0.001)
#make_optimizer(Adam, model, lr=0.0001, weight_decay=0)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=7, min_lr=0.000001)
num_epochs = 100
verbose = 1
min_val_loss = 100000
early_stop_count, early_stop_epoch = 0, 9
for i in range(num_epochs):
    if early_stop_count >= early_stop_epoch:
        break
    train_loss = 0
    model.train()
    with tqdm(total=len(train_loader)) as pbar:
        for Id, xi, yi in train_loader:
            xi[0] = xi[0].to(device)
            xi[1] = xi[1].to(device)
            yi = yi.to(torch.float32)
            yi = yi.to(device)
            y_pred = model({0:xi[0], 1:xi[1]})
            optimizer.zero_grad()

            loss = lossfun(y_pred, yi)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            pbar.set_description(f"[epoch {i+1}/{num_epochs} ]")
            pbar.set_postfix_str(f"loss = {loss.item():.4f}")
            pbar.update(1)

    val_loss = validate(device, model, val_loader)
    scheduler.step(val_loss) 
    print (f"train_loss = {train_loss/len(train_loader):.4f} val loss = {val_loss/len(val_loader):.4f}")
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(model, './model/torch_binary_covid_frozen.model')
        early_stop_count = 0
    else:
        early_stop_count += 1
