import os
import cv2
import tifffile as tif
from utils.compute_metric import compute_metric
from utils.f1_score import compute_af1_results
from utils.miou import eval_metrics
import numpy as np
import argparse
from skimage import measure, morphology
from metrics.evaluation_instance import EvaluationInstance

def get_f1_score(args):
    seg_dir = args.seg_dir
    gt_dir = args.gt_dir
    gt_suffix = args.gt_suffix
    all_f1_results = []
    for ii, seg_name in enumerate(sorted(os.listdir(seg_dir))):
        print("Calculating the {}/{} images....".format(ii, len(os.listdir(seg_dir))), end='\r')
        seg_path = os.path.join(seg_dir, seg_name)
        gt_path = os.path.join(gt_dir, seg_name.split('.')[0] + gt_suffix)
        seg = tif.imread(seg_path)
        gt = tif.imread(gt_path)
        f1_list = np.array(compute_af1_results(gt, seg, 0, 0))
        if all_f1_results==[]:
            all_f1_results = f1_list
        else:
            all_f1_results += f1_list
    all_f1_results = all_f1_results/len(os.listdir(seg_dir))
    F1_scores={
        'F1@0.5': all_f1_results[0],
        'F1@0.75': all_f1_results[5],
        'F1@0.9': all_f1_results[8],
        'F1@0.5:1.0:0.05':np.mean(all_f1_results),
    }
    
    return F1_scores

def get_mIoU_metrics(args):
    results_dir = args.seg_dir
    semantic_dir = args.semantic_dir
    num_classes = args.net_num_classes
    aAcc = []
    IoU = []
    Acc = []
    for ii, result_name in enumerate(os.listdir(results_dir)):
        print("Calculating the {}/{} results....".format(ii,len(os.listdir(results_dir))), end='\r')
        result_path = os.path.join(results_dir, result_name)
        semantic_path = os.path.join(semantic_dir, result_name.replace('.tiff','.png'))
        result = tif.imread(result_path)
        result[result>=1] = 1
        semantic = cv2.imread(semantic_path, flags=0)
        semantic[semantic==128] = 1
        ret_metrics = eval_metrics(result,
                    semantic,
                    num_classes,
                    ignore_index=255,
                    metrics=['mIoU'],
                    nan_to_num=None,
                    label_map=dict(),
                    reduce_zero_label=False,
                    beta=1)
        aAcc.append(ret_metrics['aAcc'])
        IoU.append(ret_metrics['IoU'])
        Acc.append(ret_metrics['Acc'])
        # print(ret_metrics)
    aAcc = np.mean(aAcc)
    IoU = np.mean(np.stack(IoU), axis=0)
    Acc = np.mean(np.stack(Acc), axis=0)

    return aAcc, IoU, Acc

def get_instance_metrics(args):
    dq_list, sq_list, pq_list, aji_score_list, Dice_obj_list, IoU_obj_list, Hausdorff_list = [], [], [], [], [], [], []
    seg_dir = args.seg_dir
    gt_dir = args.gt_dir
    gt_suffix = args.gt_suffix
    for ii, seg_name in enumerate(sorted(os.listdir(seg_dir))):
        print("Calculating the {}/{} images....".format(ii, len(os.listdir(seg_dir))))
        seg_path = os.path.join(seg_dir, seg_name)
        gt_path = os.path.join(gt_dir, seg_name.split('.')[0] + gt_suffix)
        seg = tif.imread(seg_path)
        gt = tif.imread(gt_path)
        metric_instance = EvaluationInstance(is_save=False, predict=seg[:,:,np.newaxis], truth=gt[:,:,np.newaxis])
        dq, sq, pq, aji_score, Dice_obj, IoU_obj, Hausdorff = metric_instance(save_path=None)
        dq_list.append(dq)
        sq_list.append(sq)
        pq_list.append(pq)
        aji_score_list.append(aji_score)
        Dice_obj_list.append(Dice_obj)
        IoU_obj_list.append(IoU_obj)
        Hausdorff_list.append(Hausdorff)
    # Instance metrics
    dq_value = np.mean(dq_list)
    sq_value = np.mean(sq_list)
    pq_value = np.mean(pq_list)
    aji_score_value = np.mean(aji_score_list)
    Dice_obj_value = np.mean(Dice_obj_list)
    IoU_obj_value = np.mean(IoU_obj_list)
    Hausdorff_value = np.mean(Hausdorff_list)
    
    return dq_value, sq_value, pq_value, aji_score_value, Dice_obj_value, IoU_obj_value, Hausdorff_value


def eval_pesudo_label(gt_dir, distmap_dir):
    distmap_names = os.listdir(distmap_dir)
    all_f1_results = []
    aAcc, IoU, Acc, Dice = [], [], [], []
    for distmap_name in distmap_names:
        distmap_path = os.path.join(distmap_dir, distmap_name)
        gt_path = os.path.join(gt_dir, distmap_name.replace('.png','.tif'))
        distmap = cv2.imread(distmap_path, flags=0)
        gt = tif.imread(gt_path)
        
        weak=distmap.copy()
        distmap[distmap==255]=0
        distmap = measure.label(distmap)
        f1_list = np.array(compute_af1_results(gt, distmap, 0, 0))

        if all_f1_results==[]:
            all_f1_results = f1_list
        else:
            all_f1_results += f1_list
            
        distmap[distmap>0]=1
        distmap = morphology.dilation(distmap, morphology.disk(1))
        gt[gt>0] = 1
        gt[weak==255]=255
        ret_metrics = eval_metrics(distmap,
                    gt,
                    2,
                    ignore_index=255,
                    metrics=['mIoU','mDice'],
                    nan_to_num=None,
                    label_map=dict(),
                    reduce_zero_label=False,
                    beta=1)
        aAcc.append(ret_metrics['aAcc'])
        IoU.append(ret_metrics['IoU'])
        Acc.append(ret_metrics['Acc'])
        Dice.append(ret_metrics['Dice'])
        
    all_f1_results = all_f1_results/len(distmap_names)
    F1_scores={
        'F1@0.5': all_f1_results[0],
        'F1@0.75': all_f1_results[5],
        'F1@0.9': all_f1_results[8],
        'F1@0.5:1.0:0.05':np.mean(all_f1_results),
    }
    aAcc = np.mean(aAcc)
    IoU = np.mean(np.stack(IoU), axis=0)
    Acc = np.mean(np.stack(Acc), axis=0)
    Dice = np.mean(np.stack(Dice), axis=0)
    print(F1_scores)
    print(aAcc)
    print(IoU)
    print(Acc)
    print(Dice)

if __name__=="__main__":
    # gt_dir = './data/MoNuSeg/train/gts'
    # distmap_dir = './data/MoNuSeg/train/labels_v5_fixed/distmap'
    # distmap_dir = './data/MoNuSeg/train/labels_v5_fixed/weak'
    # eval_pesudo_label(gt_dir, distmap_dir)
    parser = argparse.ArgumentParser("Evaluation.")
    parser.add_argument('--workspace', default='./workspace/PanNuke_ablation/Pan_Pan_unet50_cls2_1head_ep20_b32_crp256_iter0', type=str)
    parser.add_argument("--seg_dir", default='./workspace/Mo_Mo_unet50_ep500_b16_crp512_iter0_dice/results_val/postprocessed/seg', type=str)
    parser.add_argument("--gt_dir", default='./data/MoNuSeg/test/gts', type=str)
    parser.add_argument("--gt_suffix", default='.tif', type=str)
    parser.add_argument("--semantic_dir", default='./data/MoNuSeg/test/semantics', type=str)
    parser.add_argument("--net_num_classes", default=2, type=int)
    args = parser.parse_args()
    
    F1_score = get_f1_score(args)
    print("Results:")
    print("F1@0.5: {} ".format(F1_score['F1@0.5']))
    print("F1@0.75: {} ".format(F1_score['F1@0.75']))
    print("F1@0.9: {} ".format(F1_score['F1@0.9']))
    print("F1@0.5:1.0:0.05: {} ".format(F1_score['F1@0.5:1.0:0.05']))
    
    aAcc, IoU, Acc = get_mIoU_metrics(args)
    print("aAcc: {}".format(aAcc))
    print("IoU: {}, mIoU: {}".format(IoU, np.mean(IoU)))
    print("Acc: {}, mAcc: {}".format(Acc, np.mean(Acc)))

    # dq_value, sq_value, pq_value, aji_score_value, Dice_obj_value, IoU_obj_value, Hausdorff_value = get_instance_metrics(args)
    # print("dq: {}".format(dq_value))
    # print("sq: {}".format(sq_value))
    # print("pq: {}".format(pq_value))
    # print("aji_score: {}".format(aji_score_value))
    # print("Dice_obj: {}".format(Dice_obj_value))
    # print("IoU_obj: {}".format(IoU_obj_value))
    # print("Hausdorff: {}".format(Hausdorff_value))