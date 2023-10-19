import os
import cv2
import shutil
import glob
import json
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from scipy import ndimage

def read_info_meta(excel_path):
    df = pd.read_excel(excel_path)
    patient_id = str(df["ID号"][0]).zfill(10)
    # 血液信息
    leukocyte = float(df['术前白细胞'][0])
    neutrophil_percentage = float(df['术前中性细胞比例'][0])
    hemoglobin = float(df['术前血红蛋白'][0])
    platelets = float(df['术前血小板'][0])
    albumin = float(df["术前白蛋白"][0])
    ALT = float(df["术前ALT"][0])
    AST = float(df["术前AST"][0])
    TB = float(df["术前TB"][0])
    DB = float(df["术前DB"][0])
    GGT = float(df["术前GGT"][0])
    CEA = float(df["术前CEA"][0])
    CA199 = float(df["术前CA199"][0])
    if "无" in df["术前减黄"][0]:
        jianhuang = 0
    elif "PTBD" in df["术前减黄"][0]:
        jianhuang = 1
    elif "ERCP" in df["术前减黄"][0]:
        jianhuang = 2
    else:
        jianhuang = 3
    blood_input = torch.tensor([leukocyte, neutrophil_percentage, hemoglobin, platelets, albumin, ALT, AST, TB, DB, GGT, CEA, CA199, jianhuang],dtype=torch.float32)
    
    # 个人信息
    if df['性别'][0] =='女':
        gender = 0
    elif df['性别'][0] =='男':
        gender = 1
    else:
        gender = 2
    age = int(df['年龄'][0])

    if df['症状'][0] == "检查":
        symptom = 0
    elif df['症状'][0] == "黄疸":
        symptom = 1
    elif df['症状'][0] == "腹痛":
        symptom = 2
    elif df['症状'][0] == "消化道":
        symptom = 3
    else:
        symptom = 4

    smoke = 1 if df['吸烟史'][0] == '有' else 0
    diabetes = 1 if df['糖尿病史'][0] == '有' else 0
    hbbv_list = df['心脑血管病史'][0].split('、')
    hbbv = [0,0,0,0,0]
    if "高血压" in hbbv_list:
        hbbv[0] = 1
    if "冠心病" in hbbv_list:
        hbbv[1] = 1
    if "心梗" in hbbv_list:
        hbbv[2] = 1
    if "脑梗" in hbbv_list:
        hbbv[3] = 1
    if "脑出血" in hbbv_list:
        hbbv[4] = 1
    family = 0 if df['家族史'][0]=='无' else 1
    others_input = torch.tensor([gender, symptom, age, smoke, diabetes]+ hbbv + [family], dtype=torch.float32)

    # 生成描述
    if gender == 0:
        gender_prompt = ["She",'Her']
    elif gender == 1:
        gender_prompt = ["He","His"]
    else:
        gender_prompt = ["It",'Its']

    blood_des = "This is a CT image of a patient. {} blood test report is as follows: \
                {} white cell level is {}; \
                {} neutrophil percentage is {}%; \
                {} hemoglobin level is {}; \
                {} platelets level is {}; \
                {} albumin level is {}; \
                {} alanine aminotransferase level is {}; \
                {} aspartate aminotransferase level is {}; \
                {} total bilirubin level is {}; \
                {} direct bilirubin level is {}; \
                {} gamma-glutamyl transpeptidase level is {}; \
                {} carcinoembryonic antigen level is {}; \
                {} CA199 level is {}; \
                    ".format(
                        gender_prompt[1],
                        gender_prompt[1], leukocyte,
                        gender_prompt[1], int(neutrophil_percentage),
                        gender_prompt[1], int(hemoglobin),
                        gender_prompt[1], int(platelets),
                        gender_prompt[1], albumin,
                        gender_prompt[1], ALT,
                        gender_prompt[1], AST,
                        gender_prompt[1], TB,
                        gender_prompt[1], DB,
                        gender_prompt[1], GGT,
                        gender_prompt[1], CEA,
                        gender_prompt[1], CA199,
                    )
    blood_des = ' '.join(blood_des.split())
        
    if symptom == "检查":
        symptom_prompt = 'no symptom'
    elif symptom == "黄疸":
        symptom_prompt = "a jaundice symptom"
    elif symptom == "腹痛":
        symptom_prompt = "a stomach ache"
    elif symptom == "消化道":
        symptom_prompt = "a digestive tract abnormalities"
    else:
        symptom_prompt = "an unclear symptom"
        
    if smoke == 1:
        smoke_prompt = "with"
    else:
        smoke_prompt = "without"
    if diabetes == 1:
        diabetes_prompt = "with"
    else:
        diabetes_prompt = "without"
    if family =="无":
        family_prompt = "without"
    else:
        family_prompt = "with"
        
    hbbv_des = []
    if "高血压" in hbbv_list:
        hbbv_des.append("hypertension")
    if "冠心病" in hbbv_list:
        hbbv_des.append("coronary heart disease")
    if "心梗" in hbbv_list:
        hbbv_des.append("myocardial infarction")
    if "脑梗" in hbbv_list:
        hbbv_des.append("cerebral infarction")
    if "脑出血" in hbbv_list:
        hbbv_des.append("cerebral hemorrhage")
    if len(hbbv_des) == 0:
        hbbv_prompt = "no cardiovascular and cerebrovascular diseases"
    else:
        hbbv_prompt = ', '.join(hbbv_des)
        
    others_des = "{} is {} years old, {} smoking habit, {} diabetes and {} family cancer history.\
                    {} has {}.\
                    {} has {}.".format(
                gender_prompt[0], age, smoke_prompt, diabetes_prompt, family_prompt,
                gender_prompt[0], symptom_prompt,
                gender_prompt[0], hbbv_prompt)
    others_des = ' '.join(others_des.split())
    
    info_meta = {
        'patient_id': patient_id,
        "blood": blood_input, 
        "others": others_input,
        "blood_des": blood_des,
        "others_des": others_des,
        }
    return info_meta

def read_item(args, dcm_dir, excel_path):
    # Read dcm
    dcm_paths = sorted(glob.glob(os.path.join(dcm_dir,'*.dcm'),recursive=False))
    dcm_list = []
    for idx,dcm_path in enumerate(dcm_paths):
        dcm_file = pydicom.dcmread(dcm_path)
        dcm = dcm_file.pixel_array
        if idx>0 and idx<len(dcm_paths)-1:
            dcm = np.stack((pydicom.dcmread(dcm_paths[idx-1]).pixel_array,dcm,pydicom.dcmread(dcm_paths[idx+1]).pixel_array),axis=-1)
        else:
            dcm = np.stack((dcm,dcm,dcm),axis=-1)
        rescale_intercept = float(dcm_file.RescaleIntercept)
        rescale_slope = float(dcm_file.RescaleSlope)
        dcm = dcm*rescale_slope+rescale_intercept
        Win_center, Win_width = args.dcm_win_center, args.dcm_win_width
        low_window = Win_center - (Win_width/2)
        high_window = Win_center + (Win_width/2)
        dcm = np.clip(dcm, low_window, high_window).astype(np.float32)
        dcm = (dcm-low_window)/(high_window-low_window)*255
        dcm = dcm.astype(np.uint8)
        dcm_list.append(dcm)

    # Read Excels
    info_meta = read_info_meta(excel_path)

    # Merge
    img_list = []
    img_meta_list = []
    for i in range(1, len(dcm_list)-1):
        # Generate Image in Batch
        # img = np.stack((dcm_list[i-1], dcm_list[i], dcm_list[i+1]),axis=-1)
        # Resize
        img = cv2.resize(dcm_list[i], args.rescale, interpolation=cv2.INTER_LINEAR)
        H,W,C = img.shape
        # Center Crop
        h_start = round((H-args.crop_size[1])/2)
        w_start = round((W-args.crop_size[0])/2)
        h_end = h_start+args.crop_size[1]
        w_end = w_start+args.crop_size[0]
        img = img[h_start:h_end, w_start:w_end, :]
        img = ToTensor()(img)
        img_meta = {
            "patient_id": info_meta['patient_id'],
            "img_name": os.path.basename(dcm_paths[i]).split('.')[0],
            'anno_item': os.path.basename(dcm_dir),
            "blood": info_meta['blood'],
            "others": info_meta['others'],
            "blood_des": info_meta['blood_des'],
            "others_des": info_meta['others_des'],
        }

        img_list.append(img)
        img_meta_list.append(img_meta)

    return img_list, img_meta_list


def read_dcm_files(args, dcm_dir):
    dcm_paths = sorted(glob.glob(os.path.join(dcm_dir,'*.dcm'),recursive=False))
    dcm_list = []
    img_list = []
    img_name_list = []
    for idx,dcm_path in enumerate(dcm_paths):
        dcm_file = pydicom.dcmread(dcm_path)
        dcm = dcm_file.pixel_array
        rescale_intercept = float(dcm_file.RescaleIntercept)
        rescale_slope = float(dcm_file.RescaleSlope)
        dcm = dcm*rescale_slope+rescale_intercept
        Win_center, Win_width = args.dcm_win_center, args.dcm_win_width
        low_window = Win_center - (Win_width/2)
        high_window = Win_center + (Win_width/2)
        dcm = np.clip(dcm, low_window, high_window).astype(np.float32)
        dcm = (dcm-low_window)/(high_window-low_window)*255
        dcm_list.append(dcm)

    for i in range(1, len(dcm_list)-1):
        img = np.stack((dcm_list[i-1], dcm_list[i], dcm_list[i+1]),axis=-1)
        # Resize
        img = cv2.resize(img, args.rescale, interpolation=cv2.INTER_LINEAR)
        H,W,C = img.shape
        # Center Crop
        h_start = round((H-args.crop_size[1])/2)
        w_start = round((W-args.crop_size[0])/2)
        h_end = h_start+args.crop_size[1]
        w_end = w_start+args.crop_size[0]
        img = img[h_start:h_end, w_start:w_end, :]
        img = ToTensor()(img)
        img_list.append(img)

    return img_list


if __name__=="__main__":
    # read_img_meta(r'C:\Users\zby\Desktop\pan_test\excels\0000044430.xlsx')
    # read_dcm_files(r'C:\Users\zby\Desktop\pan_test\dcms\0000044430')
    pass