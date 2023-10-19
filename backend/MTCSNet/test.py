import os
import cv2
import numpy as np
from PIL import Image
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from skimage import measure
from datasets.WSCellseg import WSCellSeg

from models.unet import UNet
from models.unetplusplus import NestedUNet
from models.unet_parallel import UNet_parallel
from models.FCT import FCT
from models.CISNet import CISNet
from models.loss import DiceLoss, FocalLoss

from utils.slide_infer import slide_inference
from postprocess.postprocess import mc_distance_postprocessing, mc_distance_postprocessing_count
from utils.f1_score import compute_af1_results
from utils.miou import eval_metrics

def test_iter0(args, logger):
    test_dataset = WSCellSeg(args,mode=args.test_mode)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=1)
    
    if args.net_name.lower() == 'unet':
        print("Using Model: unet")
        model = UNet(args)
    elif args.net_name.lower() == 'unetplusplus':
        print("Using Model: unetplusplus")
        model = NestedUNet(args)
    elif args.net_name.lower() == 'unet_par':
        print("Using Model: unet_parallel")
        model = UNet_parallel(args)
    elif args.net_name.lower() in ['fct']:
        print("Using Model: FCT")
        model = FCT(args)
    elif args.net_name.lower() in ['cisnet']:
        print("Using Model: cisnet")
        model = CISNet(args)
    else:
        raise NotImplementedError("Model {} is not implemented!".format(args.net_name.lower()))

    logger.info("Loading checkpoint from: {}".format(os.path.join(args.workspace,args.checkpoint)))
    model.load_state_dict(torch.load(os.path.join(args.workspace,args.checkpoint)),strict=True)
    model = model.cuda()
    # model = torch.nn.DataParallel(model.cuda())
    model.eval()
    
    logger.info("============== Testing ==============")
    all_f1_results, vor_f1_results, heat_f1_results = [], [], []
    post_pred_f1_results, post_vor_f1_results, post_heat_f1_results = [], [], []
    count_mae, count_post_pred_f1_results, count_post_heat_f1_results = [], [], []
    aAcc, IoU, Acc, Dice, post_aAcc, post_IoU, post_Acc, post_Dice = [], [], [], [], [], [], [], []
    vor_aAcc, vor_IoU, vor_Acc, vor_Dice, vor_post_aAcc, vor_post_IoU, vor_post_Acc, vor_post_Dice = [], [], [], [], [], [], [], []
    for ii, item in enumerate(test_dataloader):
        if ii%10 == 0:
            logger.info("Testing the {}/{} images...".format(ii,len(test_dataloader)))
        imgs, annos, img_meta = item
        preds_list = []
        preds_vor_list = []
        certs_list = []
        heats_list = []
        degs_list = []
        counts_list = []
        for idx, img in enumerate(imgs):
            img = img.cuda()
            valid_region = img_meta['valid_region'][idx]
            deg_map = img_meta['deg_map'][idx].cuda()
            
            img = torch.concat((img, deg_map.unsqueeze(1)), dim=1)
            
            with torch.no_grad():
                preds, preds_vor, certs, heats, degs, counts = slide_inference(model, img, img_meta, rescale=True, args=args, valid_region=valid_region)
            # Classification
            preds_list.append(preds.detach().cpu())
            preds_vor_list.append(preds_vor.detach().cpu())
            certs_list.append(certs.detach().cpu())
            heats_list.append(heats.detach().cpu())
            degs_list.append(degs.detach().cpu())
            counts_list.append(counts.detach().cpu())
        # Fusion
        if args.test_fusion =='mean':
            fused_preds = torch.mean(torch.stack(preds_list,dim=0), dim=0)
            fused_preds_vor = torch.mean(torch.stack(preds_vor_list, dim=0), dim=0)
            fused_certs = torch.mean(torch.stack(certs_list,dim=0), dim=0)
            fused_heats = torch.mean(torch.stack(heats_list,dim=0), dim=0)
            fused_degs = torch.mean(torch.stack(degs_list,dim=0), dim=0)
            fused_counts = torch.mean(torch.stack(counts_list, dim=0), dim=0)
        if args.test_fusion == 'max':
            fused_preds,_ = torch.max(torch.stack(preds_list,dim=0), dim=0)
            fused_preds_vor,_ = torch.max(torch.stack(preds_vor_list, dim=0), dim=0)
            fused_certs,_ = torch.max(torch.stack(certs_list,dim=0), dim=0)
            fused_heats,_ = torch.max(torch.stack(heats_list,dim=0), dim=0)
            fused_degs,_ = torch.max(torch.stack(degs_list,dim=0), dim=0)
            fused_counts,_ = torch.max(torch.stack(counts_list,dim=0), dim=0)
            
        fused_preds = torch.softmax(fused_preds, dim=1)
        pred_cls = torch.argmax(fused_preds,dim=1).squeeze().numpy()
        pred_scores = fused_preds[:,1,:,:].squeeze().numpy()
        pred_vor_cls = torch.argmax(fused_preds_vor.detach().cpu(), dim=1).squeeze().numpy()
        pred_vor_scores = fused_preds_vor[:,1,:,:].squeeze().numpy()
        cert_scores = torch.sigmoid(fused_certs[:,0,:,:]).squeeze().numpy()
        heat_scores = torch.sigmoid(fused_heats[:,0,:,:]).squeeze().numpy()
        deg_scores = fused_degs[:,:,:,:].squeeze().numpy()*args.distance_scale
        
        count_scores = fused_counts.squeeze().numpy()/args.count_scale
        seed_thres = np.max(count_scores)*0.7
        seeds = count_scores>seed_thres         # can be used for
        
        # classifier head
        _, post_pred_seeds, post_pred_seg = mc_distance_postprocessing(pred_scores, args.infer_threshold, args.infer_seed, downsample=False)
        _, post_heat_seeds, post_heat_seg = mc_distance_postprocessing(heat_scores, args.infer_threshold, args.infer_seed, downsample=False)
        # vor head
        _, post_pred_seeds, post_pred_vor_seg = mc_distance_postprocessing(pred_vor_scores, args.infer_threshold, args.infer_seed, downsample=False)
        # count head
        _, post_pred_seeds, count_post_pred_seg = mc_distance_postprocessing_count(pred_scores, args.infer_threshold, seeds, downsample=False)
        _, post_heat_seeds, count_post_heat_seg = mc_distance_postprocessing_count(heat_scores, args.infer_threshold, seeds, downsample=False)
        
        # Calculate the metrics
        # F1 score from classification
        seg = np.zeros_like(pred_cls)
        seg[pred_cls==1] = 1
        seg = measure.label(seg, background=0)
        gt = img_meta['gt'].squeeze().numpy()
        f1_list = np.array(compute_af1_results(gt, seg, 0, 0))
        if all_f1_results==[]:
            all_f1_results = f1_list
        else:
            all_f1_results += f1_list
        # F1 score from vor
        vor_seg = np.zeros_like(pred_vor_cls)
        vor_seg[pred_vor_cls==1]=1
        vor_seg = measure.label(vor_seg, background=0)
        f1_list = np.array((compute_af1_results(gt, vor_seg, 0, 0)))
        if vor_f1_results==[]:
            vor_f1_results = f1_list
        else:
            vor_f1_results += f1_list
        # F1 score from heatmap
        heat_seg = np.zeros_like(heat_scores)
        heat_seg[heat_scores >= args.infer_threshold] = 1
        heat_seg[heat_scores < args.infer_threshold] = 0
        seg = measure.label(heat_seg, background=0)
        f1_list = np.array(compute_af1_results(gt, seg, 0, 0))
        if heat_f1_results==[]:
            heat_f1_results = f1_list
        else:
            heat_f1_results += f1_list
        # F1 score from post classification
        f1_list = np.array(compute_af1_results(gt, post_pred_seg, 0, 0))
        if post_pred_f1_results==[]:
            post_pred_f1_results = f1_list
        else:
            post_pred_f1_results += f1_list
        # F1 score from post vor
        f1_list = np.array(compute_af1_results(gt, post_pred_vor_seg, 0, 0))
        if post_vor_f1_results==[]:
            post_vor_f1_results = f1_list
        else:
            post_vor_f1_results += f1_list
        # F1 score from post heatmap
        f1_list = np.array(compute_af1_results(gt, post_heat_seg, 0, 0))
        if post_heat_f1_results==[]:
            post_heat_f1_results = f1_list
        else:
            post_heat_f1_results += f1_list
        # F1 score from count post classification
        f1_list = np.array(compute_af1_results(gt, count_post_pred_seg, 0, 0))
        if count_post_pred_f1_results==[]:
            count_post_pred_f1_results = f1_list
        else:
            count_post_pred_f1_results += f1_list
        # F1 score from count post heatmap
        f1_list = np.array(compute_af1_results(gt, count_post_heat_seg, 0, 0))
        if count_post_heat_f1_results==[]:
            count_post_heat_f1_results = f1_list
        else:
            count_post_heat_f1_results += f1_list
        
        # IoU
        semantic = img_meta['semantic'].squeeze().numpy()
        semantic[semantic==128] = 1
        if args.net_num_classes == 3:
            semantic[semantic==255] = 2
        semantic_new = np.zeros(semantic.shape)
        semantic_new[gt>0] = 1
        # ret_metrics = eval_metrics(pred_cls, semantic, args.net_num_classes, ignore_index=255)
        ret_metrics = eval_metrics(pred_cls, semantic_new, args.net_num_classes, ignore_index=255, metrics=['mIoU','mDice'])
        aAcc.append(ret_metrics['aAcc'])
        IoU.append(ret_metrics['IoU'])
        Acc.append(ret_metrics['Acc'])
        Dice.append(ret_metrics['Dice'])
        ret_metrics = eval_metrics(pred_vor_cls, semantic_new, args.net_num_classes, ignore_index=255, metrics=['mIoU','mDice'])
        vor_aAcc.append(ret_metrics['aAcc'])
        vor_IoU.append(ret_metrics['IoU'])
        vor_Acc.append(ret_metrics['Acc'])
        vor_Dice.append(ret_metrics['Dice'])
        
        post_pred_cls = np.zeros_like(post_pred_seg)
        post_pred_cls[post_pred_seg>0]=1
        post_pred_vor_cls = np.zeros_like(post_pred_vor_seg)
        post_pred_vor_cls[post_pred_vor_seg>0]=1
        ret_metrics = eval_metrics(post_pred_cls, semantic_new, args.net_num_classes, ignore_index=255, metrics=['mIoU','mDice'])
        print(ret_metrics['IoU'])
        if ret_metrics['IoU'][1] == np.NaN:
            print(np.unique(gt), np.sum(gt>=1), np.sum(post_pred_cls>=1))
            print(img_meta['img_path'][0])
            
        post_aAcc.append(ret_metrics['aAcc'])
        post_IoU.append(ret_metrics['IoU'])
        post_Acc.append(ret_metrics['Acc'])
        post_Dice.append(ret_metrics['Dice'])
        ret_metrics = eval_metrics(post_pred_vor_cls, semantic_new, args.net_num_classes, ignore_index=255, metrics=['mIoU','mDice'])
        vor_post_aAcc.append(ret_metrics['aAcc'])
        vor_post_IoU.append(ret_metrics['IoU'])
        vor_post_Acc.append(ret_metrics['Acc'])
        vor_post_Dice.append(ret_metrics['Dice'])
        
        # Visualization
        pred_cls[pred_cls==1] = 128
        pred_cls[pred_cls==2] = 255
        pred_vor_cls[pred_vor_cls==1]=128
        pred_vor_cls[pred_vor_cls==2]=255
        count_scores = count_scores/np.max(count_scores)*255
        img_name = os.path.basename(img_meta['img_path'][0]).split('.')[0]+'.png'
        save_path = os.path.join(args.workspace,args.results_test,'pred', img_name)
        save_score_path = os.path.join(args.workspace,args.results_test, 'score', img_name)
        vor_save_path = os.path.join(args.workspace, args.results_val,'vor', img_name)
        vor_score_save_path = os.path.join(args.workspace, args.results_val,'vor_score', img_name)
        heat_save_path = os.path.join(args.workspace, args.results_test, 'heat', img_name)
        deg_save_path = os.path.join(args.workspace, args.results_test,'deg', img_name.replace('.png','_deg{}.pkl'.format(args.test_degree)))
        count_save_path = os.path.join(args.workspace, args.results_test,'count', img_name)
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        os.makedirs(os.path.dirname(save_score_path),exist_ok=True)
        os.makedirs(os.path.dirname(vor_save_path),exist_ok=True)
        os.makedirs(os.path.dirname(vor_score_save_path),exist_ok=True)
        os.makedirs(os.path.dirname(heat_save_path),exist_ok=True)
        os.makedirs(os.path.dirname(deg_save_path),exist_ok=True)
        os.makedirs(os.path.dirname(count_save_path),exist_ok=True)
        cv2.imwrite(save_path, pred_cls)
        cv2.imwrite(save_score_path, pred_scores*255)
        cv2.imwrite(vor_save_path, pred_vor_cls)
        cv2.imwrite(vor_score_save_path, pred_vor_scores*255)
        cv2.imwrite(heat_save_path, heat_scores*255)
        if args.net_degree:
            pickle.dump(deg_scores, open(deg_save_path, 'wb'))
        cv2.imwrite(count_save_path, count_scores)
        
    logger.info("Test Complete!!!")

if __name__=="__main__":
    test_iter0()