import cv2
import tifffile as tif
import os
import numpy as np
import glob

def statistic_cpm(data_dir):
    img_paths = glob.glob(os.path.join(data_dir, '**/image??.png'),recursive=True)
    H_list = []
    W_list = []
    total_cells = 0
    for ii, img_path in enumerate(img_paths):
        print("Processing the {}/{} images...".format(ii, len(img_paths)),end='\r')
        gt_path = img_path.replace('.png', '_mask.png').replace('segmentation-test-images','segmentation-test-masks').replace('test-images/','')
        
        img = cv2.imread(img_path)
        gt = cv2.imread(gt_path)
        gt = gt[:,:,2]
        
        H,W =img.shape[:2]
        num_cell = np.max(gt)
        print(gt_path, num_cell)
        H_list.append(H)
        W_list.append(W)
        total_cells += num_cell
        
    print("Max H:{}, Max W:{}, Min H:{}, Min W:{}, cell num:{}".format(max(H_list),max(W_list),min(H_list),min(W_list), total_cells))
        


if __name__=="__main__":
    # data_dir = './data/CPM2017/training/segmentation_training'
    data_dir = './data/CPM2017/test/segmentation-test'
    statistic_cpm(data_dir)
    