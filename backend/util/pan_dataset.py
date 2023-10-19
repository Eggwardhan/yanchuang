import torch
from torch.utils.data import Dataset,DataLoader
import PANCLS.models.text_encoder.clip as clip

class PanDataset(Dataset):
    def __init__(self, list1, list2):
        self.list1 = list1
        self.list2 = list2
    
    def __len__(self):
        return len(self.list1)
    
    def __getitem__(self, index):
        return self.list1[index] , self.list2[index]
        img_meta=self.list2[index]
        
        return  {
            'img': self.list1[index],
            "blood": img_meta['blood'],
            'others': img_meta['others'],
            "blood_des": clip.tokenize(img_meta['blood_des'], context_length=256),
            "others_des": clip.tokenize(img_meta['others_des']),
            "anno_item":img_meta['anno_item'],
            "img_name":img_meta['img_name']
            
                }
    

def collate_fn(batch):
    img_batch,meta_batch = zip(*batch)
    # blood_batch = [meta['blood'] for meta in meta_batch]        
    blood_batch = meta_batch[0]['blood']
    other_batch = meta_batch[0]['others']
    
    blood_des_batch = [meta['blood_des'] for meta in meta_batch]
    others_des_batch = [meta['others_des'] for meta in meta_batch]
    
    anno_batch = [meta['anno_item'] for meta in meta_batch]
    name_batch = [meta['img_name'] for meta in meta_batch]
    
    return {
            'img': torch.stack(img_batch).cuda(),
            "blood": blood_batch.cuda(),
            'others': other_batch.cuda(),
            "blood_des": clip.tokenize(blood_des_batch, context_length=256).cuda(),
            "others_des": clip.tokenize(others_des_batch).cuda(),
            "anno_item":anno_batch,
            "img_name":name_batch
            
                }