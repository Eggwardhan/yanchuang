import json


if __name__=="__main__":
    split_paths = ['data/splits/splits_10fold/TNBC/train_tnbc_val_tnbc_fold{}.json'.format(i) for i in range(1,11)]
    split_dict = {'train':[], 'val':[]}
    for ii, split_path in enumerate(split_paths):
        fold_dict = json.load(open(split_path,'r'))
        if ii in [0]:
            split_dict['val'] += fold_dict['val']
        else:
            split_dict['train'] += fold_dict['val']
            
    json.dump(split_dict, open('data/splits/train_tnbc_val_tnbc.json','w'), indent=2)