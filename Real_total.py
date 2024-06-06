import streamlit as st
from PIL import Image
from io import BytesIO
import base64
import nibabel as nib
import numpy as np
import shutil
import monai
import torch
import matplotlib.pyplot as plt
import gc
import time
import streamlit_ext as ste
import os
import pyvista as pv
from stpyvista import stpyvista
import xlsxwriter
from io import BytesIO
import glob
import subprocess


    
st.set_page_config(layout="wide", page_title="BreastMRI", page_icon=":ribbon:")
st.write("## Segmentation from T1")
st.write("3D T1으로부터 Segmentation")
st.sidebar.write("## Upload Input Files :tulip:")


def fix_image(upload):   
    
    nw = time
    mystr = nw.strftime('%Y-%m-%d %H:%M:%S')
    st.write(f"- {mystr}: fgt segmentation start")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.SwinUNETR(img_size=(96,96,32), in_channels=1, out_channels=3, feature_size=48).to(device)
    model.load_state_dict(torch.load("/home/hufsaim/Python/2024_CapStone/software/jihyun/model/fgtseg/seg_model_last_v0813.pth", map_location = torch.device(device)))
    model.eval()
    
    h = nib.load(upload)
    img = torch.tensor(h.get_fdata())
    sc = monai.transforms.ScaleIntensity(minv=0,maxv=1.0) #object
    img = sc(img)
    img = img.unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = monai.inferers.sliding_window_inference(img, [96,96,32], 4, model, overlap=0.25,mode='gaussian')
        
    out = out.softmax(1)
    model.cpu()
    
    del model
    gc.collect()
    torch.cuda.empty_cache()    
    r = torch.argmax(out,axis=1)
    r = r[0].clone().detach().cpu().numpy() 
    msk = r.copy()
    zc = int(r.shape[-1]*0.5)
    
    nw = time
    mystr = nw.strftime('%Y-%m-%d %H:%M:%S')
    st.write(f"- {mystr}: segmentation end")

    
    #streamlit
    col1.write("### [Result]")
    fig1, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3,sharey=True)
    ax1.set_title('Original')
    ax1.imshow(np.rot90(np.min(img[0,0,:,:,zc-2:zc+2].clone().detach().cpu().numpy(),axis=-1)),cmap='gray')
    ax1.axis('off')
    ax2.set_title('Breast/FGT')
    ax2.imshow(np.rot90(np.max(r[:,:,zc-2:zc+2],axis=-1)),cmap='gray')
    ax2.axis('off')
    ax3.set_title('Z-axis sum')
    ax3.imshow(np.rot90(np.sum(r,axis=-1)),cmap='gray')
    ax3.axis('off')
    col1.pyplot(fig1)

    
    r = nib.Nifti1Image((r).astype(np.int16), h.affine)
    nib.save(r,'/home/hufsaim/Python/2024_CapStone/software/jihyun/tmp/tmp.nii.gz')
    st.sidebar.markdown("\n")
    
    with open("/home/hufsaim/Python/2024_CapStone/software/jihyun/tmp/tmp.nii.gz", "rb") as fp:
        btn = ste.download_button(
            label="Download segmentation result",
            data=fp,
            file_name="fgtseg.nii.gz",
        )
    torch.cuda.empty_cache()
    gc.collect()
    return msk, h




def measurements (f_, f):
    cnt_b = np.sum(f_ == 1)
    cnt_f = np.sum(f_ == 2)
    sx, sy, sz =f.header.get_zooms()
    vol = sx * sy * sz
    vol_b = cnt_b * vol
    vol_f = cnt_f * vol
    ccvol_b = vol_b * 0.001
    ccvol_f = vol_f * 0.001

#     voxel_size = f.header.get_zooms()
#     voxel_volume = np.prod(voxel_size)

#     # 각 레이블의 부피를 계산합니다.
#     unique_values, counts = np.unique(f_, return_counts=True)
#     volumes = {value: count * voxel_volume for value, count in zip(unique_values, counts)}
#     volumes_cc = {value: volume / 1000 for value, volume in volumes.items()}

#     # 레이블 1 (유방 부피)와 레이블 2 (FGT 부피)를 찾고 부피를 계산합니다.
#     ccvol_b = volumes_cc.get(1, 0.0)
#     ccvol_f = volumes_cc.get(2, 0.0)
    
    col2.write("### [Measurements]")
    col2.write(f"- Breast volume : {ccvol_b:.2f} (cc)")
    col2.write(f"- FGT volume : {ccvol_f:.2f} (cc)")
    
    output = BytesIO()
    workbook = xlsxwriter.Workbook(output, {'in_memory': True})
    worksheet = workbook.add_worksheet()
    worksheet.write('A1', 'pid')
#     worksheet.write('A2', '')
    worksheet.write('B1', 'breast')
    worksheet.write('B2', ccvol_b)
    worksheet.write('C1', 'fgt')
    worksheet.write('C2', ccvol_f)
    workbook.close()
    with col2:
        st.download_button(
            label="Download Excel",
            data=output.getvalue(),
            file_name="measurement.xlsx",
            mime="application/vnd.ms-excel"
        )
        
# def measurements (f_, f):
#     margin = 10
#     breast_region = np.where(f_ == 1)
#     fgt_region = np.where(f_ == 2)

#     if len(breast_region[0]) == 0 or len(fgt_region[0]) == 0:
#         raise ValueError("유방 또는 FGT 영역을 찾을 수 없습니다.")

#     # 유방 영역의 최소 및 최대 좌표 계산
#     x_min, x_max = breast_region[0].min(), breast_region[0].max()
#     y_min, y_max = breast_region[1].min(), breast_region[1].max()
#     z_min, z_max = breast_region[2].min(), breast_region[2].max()

#     # 바운딩 박스를 기준으로 일정 범위 외부의 가장자리 제거
#     f_[0:max(0, x_min - margin), :, :] = 0
#     f_[min(f_.shape[0], x_max + margin):, :, :] = 0
#     f_[:, 0:max(0, y_min - margin), :] = 0
#     f_[:, min(f_.shape[1], y_max + margin):, :] = 0
#     f_[:, :, 0:max(0, z_min - margin)] = 0
#     f_[:, :, min(f_.shape[2], z_max + margin):] = 0

#     # 볼륨 계산
#     cnt_b = np.sum(f_ == 1)
#     cnt_f = np.sum(f_ == 2)
#     sx, sy, sz = f.header.get_zooms()
#     vol = sx * sy * sz
#     vol_b = cnt_b * vol
#     vol_f = cnt_f * vol
#     ccvol_b = vol_b * 0.001
#     ccvol_f = vol_f * 0.001

#     # 결과 출력
#     col2.write("### [Measurements]")
#     col2.write(f"- Breast volume : {ccvol_b:.2f} (cc)")
#     col2.write(f"- FGT volume : {ccvol_f:.2f} (cc)")

#     # 엑셀 파일 생성
#     output = BytesIO()
#     workbook = xlsxwriter.Workbook(output, {'in_memory': True})
#     worksheet = workbook.add_worksheet()
#     worksheet.write('A1', 'pid')
#     worksheet.write('B1', 'breast')
#     worksheet.write('B2', ccvol_b)
#     worksheet.write('C1', 'fgt')
#     worksheet.write('C2', ccvol_f)
#     workbook.close()

#     with col2:
#         st.download_button(
#             label="Download Excel",
#             data=output.getvalue(),
#             file_name="measurement.xlsx",
#             mime="application/vnd.ms-excel"
#         )

def visual_mask(msk):
    
    st.write("### [3D Rendering]")
    nw = time
    mystr = nw.strftime('%Y-%m-%d %H:%M:%S')
    st.write(f"- {mystr}: rendering start")
    
    msk = msk.astype(np.int16)
    plotter = pv.Plotter()
    plotter.background_color = "black"
    grid = pv.ImageData()
    grid.dimensions = np.array(msk.shape) + 1  # Z축 크기를 1로 설정
    grid.spacing = (1, 1, 1)  # 필요에 따라 간격 조정 가능
    grid.cell_data["seg"] = msk.flatten(order="F")
    plotter.add_volume(grid, cmap="coolwarm", opacity="sigmoid")  # 색상 맵 및 투명도 조정 가능
    stpyvista(plotter)
    nw = time
    mystr = nw.strftime('%Y-%m-%d %H:%M:%S')
    st.write(f"- {mystr}: rendering end")
    
def feedback (f, mask):
    
    number = st.text_input("Assign a folder number before clicking feedback button.")
    st.write(number)

    if st.button(":+1: good"):
        
        mydir = f'/home/hufsaim/Python/2024_CapStone/software/jihyun/update_data/good/{number}' #실행된 시점
        
        
        if os.path.exists(mydir) == False:
            os.mkdir(mydir)
            
            nib.save(f, os.path.join(mydir, 'original.nii.gz'))
            mask = nib.Nifti1Image((mask).astype(np.int16), f.affine)
            nib.save(mask, os.path.join(mydir, 'mask.nii.gz'))
        
        st.write("Thank you for your feedback!")
        
    elif st.button(":-1: bad"):
        
        mydir = f'/home/hufsaim/Python/2024_CapStone/software/jihyun/update_data/bad/{number}'
        
        if os.path.exists(mydir) == False:
            os.mkdir(mydir)
            
        nib.save(f, os.path.join(mydir, 'original.nii.gz'))
        mask = nib.Nifti1Image((mask).astype(np.int16), f.affine)
        nib.save(mask, os.path.join(mydir, 'mask.nii.gz'))
        
        st.write("Thank you for your feedback!")
        
 


    
def check ():
    if not (os.path.isdir('Python/2024_CapStone/software/jihyun/update_data/good')):
        return 0
    original = glob.glob('Python/2024_CapStone/software/jihyun/update_data/good/*')
    
    if (len(original) >= 30):
        return 1
    else: return 0



col1,col2 = st.columns(2)
my_upload1 = st.sidebar.file_uploader("Upload a T1 image (nii.gz)", type=["nii.gz"])
my_upload2 = st.sidebar.file_uploader("Upload a T1 image (nii)", type=["nii"])


if my_upload1:
    with open("tmp.nii.gz","wb") as buffer:
        shutil.copyfileobj(my_upload1, buffer)
    mask, h = fix_image("tmp.nii.gz")
    measurements(mask, h)
    visual_mask(mask)
    feedback(h, mask)
#     processed_npmsk = removing_FP(mask)
    
#     val = check()
    
#     try:
#         result = os.system('python3 Python/2024_CapStone/software/jihyun/update_open_code.py')
#         print("Script executed with exit code:", result)
#     except Exception as e:
#         print("An error occurred:", e)
        
#     try:
#         result = subprocess.run(['python3', 'Python/2024_CapStone/software/jihyun/update_code.py'], check=True, capture_output=True, text=True)
        
#         result = os.system(['python3',os.system('update_code.py')
#         print("Script output:", result.stdout)
        
#     except subprocess.CalledProcessError as e:
#         print("Error executing script:", e.stderr)



#     if (val == 1):
#         os.system('Python/2024_CapStone/software/jihyun/Untitled-Copy1.ipynb')


elif my_upload2:
    with open("tmp.nii","wb") as buffer:
        shutil.copyfileobj(my_upload2, buffer)
    mask, h = fix_image("tmp.nii")
    measurements(mask, h)
    visual_mask(mask)
    feedback(h, mask)
#     processed_npmsk = removing_FP(mask)
    
#     val = check()
#     try:
#         result = os.system('python3 Python/2024_CapStone/software/jihyun/update_open_code.py')
#         print("Script executed with exit code:", result)
#     except Exception as e:
#         print("An error occurred:", e)
        
        
        
        
#     try:
#         result = subprocess.run(['python3', 'Python/2024_CapStone/software/jihyun/update_code.py'], check=True, capture_output=True, text=True)
#         print("Script output:", result.stdout)
        
#     except subprocess.CalledProcessError as e:
#         print("Error executing script:", e.stderr)


