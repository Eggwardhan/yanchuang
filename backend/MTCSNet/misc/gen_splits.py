import os
from random import sample
import json
import csv

def gen_PanNuke_split(data_root, output_path):
    img_names = sorted(os.listdir(os.path.join(data_root,'images')))
    split_dict = {'train':[],'val':[]}
    item_list = []
    for ii,img_name in enumerate(img_names):
        item_dict = {
            'img_path': os.path.join(data_root,'images',img_name),
            "gt_path": os.path.join(data_root,'gts', img_name.replace('.png','.tiff')),
            "full_label_path": os.path.join(data_root,'labels/full',img_name),
            "full_heat_path": os.path.join(data_root,'labels/full_heatmap',img_name),
            "full_dist_path": os.path.join(data_root,'labels/full_distmap',img_name),
            "weak_label_path": os.path.join(data_root,'labels/weak',img_name),
            "weak_heat_path": os.path.join(data_root,'labels/weak_heatmap',img_name),
            "weak_dist_path": os.path.join(data_root,'labels/weak_distmap',img_name),
            "semantic_path": os.path.join(data_root,'semantics',img_name),
            "point_path": os.path.join(data_root,'points',img_name.replace('.png','.json')),
        }
        item_list.append(item_dict)
    test_num = int(len(item_list)*0.1)
    test_list = sample(range(len(item_list)),test_num)
    for item_idx in range(len(item_list)):
        item = item_list[item_idx]
        if item_idx in test_list:
            split_dict['val'].append(item)
        else:
            split_dict['train'].append(item)
    json.dump(split_dict, open(output_path,'w'), indent=2)

def gen_Lizard_split(data_root, info_csv, output_path):
    csv_file = csv.reader(open(info_csv,'r'))
    split_fold1, split_fold2, split_fold3 = [], [], []
    for row in csv_file:
        print(row)
        img_name, source, split_idx = row
        item_dict = {
            'img_path': os.path.join(data_root,'images',img_name+'.png'),
            "gt_path": os.path.join(data_root,'gts', img_name+'.tiff'),
            "full_label_path": os.path.join(data_root,'labels/full',img_name+'.png'),
            "full_heat_path": os.path.join(data_root,'labels/full_heatmap',img_name+'.png'),
            "full_dist_path": os.path.join(data_root,'labels/full_distmap',img_name+'.png'),
            "weak_label_path": os.path.join(data_root,'labels/weak',img_name+'.png'),
            "weak_heat_path": os.path.join(data_root,'labels/weak_heatmap',img_name+'.png'),
            "weak_dist_path": os.path.join(data_root,'labels/weak_distmap',img_name+'.png'),
            "semantic_path": os.path.join(data_root,'semantics',img_name+'.png'),
            "point_path": os.path.join(data_root,'points',img_name+'.json'),
        }
        if split_idx == '1':
            split_fold1.append(item_dict)
        elif split_idx == '2':
            split_fold2.append(item_dict)
        elif split_idx == '3':
            split_fold3.append(item_dict)
        else:
            pass
    split_3 = {'train':split_fold1+split_fold2, 'val':split_fold3}
    split_2 = {'train':split_fold1+split_fold3, 'val':split_fold2}
    split_1 = {'train':split_fold2+split_fold3, 'val':split_fold1}
    
    json.dump(split_1, open(os.path.join(output_path.replace('.json','_fold1.json')),'w'), indent=2)
    json.dump(split_2, open(os.path.join(output_path.replace('.json','_fold2.json')),'w'), indent=2)
    json.dump(split_3, open(os.path.join(output_path.replace('.json','_fold3.json')),'w'), indent=2)
        
def gen_All_split(split_path_list, output_path):
    all_split_dict = {'train':[],'val':[]}
    for split_path in split_path_list:
        split_dict = json.load(open(split_path,'r'))
        all_split_dict['train'] += split_dict['train']+split_dict['val']
        all_split_dict['val'] += split_dict['val']
        
    json.dump(all_split_dict, open(output_path,'w'), indent=2)
        
if __name__=="__main__":
    # data_root = './data/PanNuke'
    # output_path = 'data/splits/train_pan_val_pan.json'
    # gen_PanNuke_split(data_root, output_path)
    
    # data_root = './data/Lizard'
    # info_csv = './data/Lizard/origin/Lizard_Labels/info.csv'
    # output_path = './data/splits/lizard/train_liz_val_liz.json'
    # gen_Lizard_split(data_root, info_csv, output_path)
    
    split_path_list = ['data/splits/train_cpm_val_cpm.json',
                       'data/splits/train_cs_val_cs.json',
                       'data/splits/train_mo_val_mo.json',
                    #    'data/splits/train_pan_val_pan.json',
                       'data/splits/train_tnbc_val_tnbc.json']
    output_path = './data/splits/train_all_val_all.json'
    gen_All_split(split_path_list, output_path)