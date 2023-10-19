import cv2
import tifffile as tif
import os
import numpy as np

def statistic_tnbc(img_dir, gt_dir):
    img_names = os.listdir(img_dir)
    H_list = []
    W_list = []
    total_cells = 0
    for ii, img_name in enumerate(img_names):
        print("Processing the {}/{} images...".format(ii, len(img_names)),end='\r')
        img_path = os.path.join(img_dir, img_name)
        gt_path = os.path.join(gt_dir, img_name.replace('.png','.tiff'))
        
        img = cv2.imread(img_path)
        gt = tif.imread(gt_path)
        
        H,W,C =img.shape
        num_cell = np.max(gt)
        H_list.append(H)
        W_list.append(W)
        total_cells += num_cell
        
    print("Max H:{}, Max W:{}, Min H:{}, Min W:{}, cell num:{}".format(max(H_list),max(W_list),min(H_list),min(W_list), total_cells))
        
    
    
if __name__=="__main__":
    img_dir = './data/TNBC/images'
    gt_dir = './data/TNBC/gts'
    statistic_tnbc(img_dir, gt_dir)
    