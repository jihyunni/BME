import os
import shutil
import tempfile
import streamlit as st
from PIL import Image
import glob
import torch
import gc
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from monai.networks.nets import SwinUNETR


import streamlit_ext as ste
import pyvista as pv
from stpyvista import stpyvista
import xlsxwriter
from io import BytesIO
import logging
from datetime import datetime

import monai
import re
import base64
import nibabel as nib
import random
from monai.utils import set_determinism, first

from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    CenterSpatialCropd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,    
    RandCropByLabelClassesd,

    RandSpatialCropSamplesd,
    RandShiftIntensityd,
    RandZoomd,
    ScaleIntensityd,
    Spacingd,
    SpatialPadd,
    GaussianSmoothd,
    RandRotate90d,
    RandRotated,
    ToTensord,
    RandSpatialCropd,
    RandGaussianSmoothd,    
    RandGaussianSharpend,
    RandGaussianNoised,
)
from monai.config import print_config

from monai.data import (
    DataLoader,
    Dataset,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)

from skimage import measure
from sklearn.metrics import roc_auc_score
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
print_config()


path = glob.glob('Python/2024_CapStone/software/jihyun/update_model/version_*/model_V*.pth')
print(path)


latest_version = max([int(file.split('_V')[-1].split('.')[0]) for file in path])
print(latest_version)

log_dir = (f'Python/2024_CapStone/software/jihyun/update_model/version_{latest_version+1}')
    
print(log_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
        
log_file = os.path.join(log_dir, f"log_V{latest_version+1}.log")

print(log_file)

# 로깅 기본 설정
logging.basicConfig(
    level=logging.INFO,  # 로깅 레벨 설정
    format="%(asctime)s - %(levelname)s - %(message)s",  # 로그 메시지 형식
    datefmt="%Y-%m-%d %H:%M:%S",  # 날짜 형식
    handlers=[
    logging.FileHandler(log_file),  # 파일 핸들러: 로그를 파일에 저장
    logging.StreamHandler()         # 스트림 핸들러: 로그를 표준 출력에 전달
    ]
)


# 로그 기록 시작
logging.info("새로운 로그 파일 생성 완료, 로그 시작")
print()

initial_model = "Python/2024_CapStone/software/jihyun/update_model/version_0/model_V0.pth"
print("initial path being searched:", initial_model)
print()

# version_dirs = glob.glob(path)
# print("Files found:", version_dirs)

# version_dirs가 비어 있지 않은 경우에만 최신 버전 계산
if path:
    latest_version = max([int(file.split('_V')[-1].split('.')[0]) for file in path])
    latest_model = f"Python/2024_CapStone/software/jihyun/update_model/version_{latest_version}/model_V{latest_version}.pth"
else:
    # version_dirs가 비어 있으면 초기 모델을 사용
    latest_model = initial_model

# 최신 모델과 초기 모델 비교
if initial_model == latest_model:
    print("Returning initial model.")
    print(initial_model)
else:
    print("Returning latest model.")
    print(latest_model)

    
data_dir = 'Python/2024_CapStone/software/jihyun/update_data/good'

all= os.listdir(data_dir)
    
good_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f)) and f != '.ipynb_checkpoints' and not re.match(r'^V', f)]
    
new_data = good_folders[:24]
print(new_data)
print()
    
S = glob.glob("/mnt/breast/data_vendor/*_S_*/fgt.nii.gz")
P = glob.glob("/mnt/breast/data_vendor/*_P_*/fgt.nii.gz")
G = glob.glob("/mnt/breast/data_vendor/*_G_*/fgt.nii.gz")

S = S[-13::-1]
random.shuffle(S)

P = P[-13::-1]
random.shuffle(P)

G = G[-13::-1]
random.shuffle(G)


past_data = S[:12] + P[:12] + G[:10]  

olds = []
for tmp in past_data:
    olds.append(tmp.split('fgt.nii.gz')[0])
print(olds)
print()

data_dicts = []
for good_files in good_folders:
    image_path = os.path.join(data_dir, good_files, 'original.nii.gz')
    label_path = os.path.join(data_dir, good_files, 'mask.nii.gz')
    
    data_dicts.append({'image': image_path, 'seg': label_path})
    
for idx, old in enumerate(olds):
    
    data_dicts.append({'image': os.path.join(olds[idx],"t1_pre.nii.gz"), "seg": os.path.join(olds[idx], "fgt.nii.gz")})
    
    
for i in data_dicts:
    print(i)
    print()                      
     
train_Data = data_dicts  


train_transforms = Compose(
    [
        LoadImaged(keys=["image","seg"]),
        EnsureChannelFirstd(keys=["image","seg"]),
        
        ScaleIntensityd(
                keys=["image"],
                minv=0.0,
                maxv=1.0,
        ),    
        
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.05,
            prob=0.5,
        ),       
        
        RandZoomd(
            keys=["image", "seg"],
            prob = 0.75,
            min_zoom = 0.5,
            max_zoom = 2.0,
            mode = ['area','nearest']
        ),

        RandGaussianNoised(keys=["image"],
                           prob=0.5, mean=0.0, std=0.01, 
                           allow_missing_keys=False),
        RandGaussianSmoothd(keys=["image"],
                            sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), sigma_z=(0.25, 1.5)
                            , approx='erf', prob=0.5, allow_missing_keys=False),
        RandGaussianSharpend(keys=["image"],
                             sigma1_x=(0.5, 1.0), sigma1_y=(0.5, 1.0), sigma1_z=(0.5, 1.0), 
                             sigma2_x=0.5, sigma2_y=0.5, sigma2_z=0.5, alpha=(10.0, 30.0)
                             , approx='erf', prob=0.5, allow_missing_keys=False),

        SpatialPadd(keys=["image","seg"], spatial_size=(192,192, 64)),
        RandCropByPosNegLabeld(
            keys=["image","seg"],
            label_key="seg",
            spatial_size=(160, 160, 48),
            pos=2,
            neg=1,
            num_samples=3,
        ),
        
        RandRotated(keys=["image","seg"],
            mode=["bilinear","nearest"],
            range_x=0.75, range_y=0.0, range_z=0.0,
            prob=0.75),
        
        CenterSpatialCropd(keys=["image","seg"],
                         roi_size=(96,96,32)),

        RandFlipd(
            keys=["image", "seg"],
            spatial_axis=[0],
            prob=0.5,
        ),
        RandFlipd(
            keys=["image","seg"],
            spatial_axis=[1],
            prob=0.5,
        ),
        RandFlipd(
            keys=["image", "seg"],
            spatial_axis=[2],
            prob=0.5,
        ),
        RandRotate90d(
            keys=["image", "seg"],
            prob=0.25,
            max_k=3,
        ),
        ToTensord(keys=["image","seg"]),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image","seg"]),
        EnsureChannelFirstd(keys=["image","seg"]),         
        ScaleIntensityd(
                keys=["image"],
                minv=0.0,
                maxv=1.0,
        ),         
        ToTensord(keys=["image","seg"]),
    ]
)
train_ds = CacheDataset(data=train_Data, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)


device = "cuda:0"
model = monai.networks.nets.SwinUNETR(img_size=(96, 96, 32), in_channels=1, out_channels=3, feature_size=48).to(device)
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
torch.backends.cudnn.benchmark = True


def train(global_step, train_loader):
    model.train()
    epoch_loss = 0
    step = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x, y = (batch["image"]).to(device), batch["seg"].to(device)
        #y[y>0] = 1
        
        logit_map = model(x)
        
        loss = loss_function(logit_map, y)
        loss.backward()
        epoch_loss += loss.item()
        
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss)
        )

        global_step += 1

    return global_step
        
        
max_iterations = 114800
eval_num = 2
post_label = AsDiscrete(to_onehot=3)
post_pred = AsDiscrete(argmax=True, to_onehot=3)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
dice_val_best = 0.0
global_step = 0
global_step_best = 0
epoch_loss_values = []
metric_values = []



optimizer = torch.optim.AdamW(model.parameters(), lr=.42e-4)

while global_step < max_iterations:
    global_step = train(
        global_step, train_loader
    )
    
torch.save(
                    model.state_dict(), os.path.join(log_dir, f"model_V{latest_version+1}.pth")
                )


new_good_path = os.path.join(data_dir, f'V{latest_version + 1}')
                             

# 새로운 폴더 생성
os.makedirs(new_good_path, exist_ok=True)
print(f"Created new folder: {new_good_path}")

# good_folders의 지정한 개수만큼의 폴더를 새로운 폴더로 이동
for folder in new_data:
    src_path = os.path.join(data_dir, folder)
    dst_path = os.path.join(new_good_path, folder)
    shutil.move(src_path, dst_path)
    print(f"Moved {folder} to {new_good_path}")
