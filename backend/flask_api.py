import os
import base64
import io
import torch
from torch.utils.data import DataLoader
from torchvision import modelsh
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify
import sys
import yaml
from shutil import copyfile
import argparse
import numpy as np

sys.path.append("/home/dmt218/zby/PANCLS")
from datasets.cellseg import CellSeg
from utils.logger import Logger
from models.custom_model_2d import Custom_Model_2D
from utils.slide_infer import slide_inference
import models.text_encoder.clip as clip

sys.path.append(0,"/home/dmt218/zby/MTCSNet")
from yacs.config import CfgNode
from datasets.WSCellseg import WSCellSeg
from models.unetplusplus import NestedUNet as NestedUNet2
from utils.slide_infer import slide_inference_mtcs
from postprocess.postprocess import mc_distance_postprocessing, mc_distance_postprocessing_count


args="/home/dmt218/hsh/yanchuang/backend/utils/eval_pancls.yaml"
app = Flask(__name__)
def get_config(path ):
    cfg = yaml.load(open(path,'r'), Loader=yaml.FullLoader)
    cfg = CfgNode(cfg)
    os.makedirs(cfg.workspace,exist_ok=True)
    os.makedirs(os.path.join(cfg.workspace, cfg.results_val), exist_ok=True)
    os.makedirs(os.path.join(cfg.workspace, cfg.results_test), exist_ok=True)
    if args.train_pass == True:
        copyfile(args.config, os.path.join(cfg.workspace, os.path.basename(args.config)))
    logger =Logger(cfg)
    logger.info(cfg)
    return cfg,logger

args , logger = get_config(args)
# 加载训练好的图像分割模型
# modelPancls = NestedUNet(args)  # 定义了一个合适的模型类
# modelPancls = torch.load("/home/dmt218/zby/PANCLS/workspace/2D_final/res50_mlp_trans_iter3000_img_b16_rz560_crp512_ecs_d0.5/best_model.pth")  # 替换成你的模型路径
# modelPancls.eval()
modelCellseg =NestedUNet2(args,mode = args.mode)
modelCellseg = torch.load("/home/dmt218/zby/MTCSNet/workspace/All/All_All_unetpp50rvdc_cls2_1head_ep200_b8_crp512_full_cn/best_model.pth")
modelCellseg.eval()

# pancls_dataset= CellSeg(args,logger)
# pancls_dataloader = DataLoader(pancls_dataset)
mtcs_dataset = WSCellSeg(args)
mtcs_dataloader= DataLoader(mtcs_dataset,batch_size= 1,num_workers=1)



# 图像预处理函数
def preprocess_pancls_image(image):
    
 
    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    image = transform(image).unsqueeze(0)
    return image

# 图像分割函数
def segment_mtcs_image(model,dloader):
    for ii, item in enumerate(dloader):
        if ii%10 == 0:
            logger.info("Testing the {}/{} images...".format(ii,len(dloader)))
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
        
def segment_pancls_image(model,dloader,test_fusion='mean'):
    for ii, item in enumerate(dloader):
        pred_invade_dict, pred_surgery_dict, pred_essential_dict = {}, {}, {}
        score_invade_dict, score_surgery_dict, score_essential_dict = {}, {}, {}
        label_invade_dict, label_surgery_dict, label_essential_dict = {}, {}, {}
        feat_dict = {}        
        with torch.no_grad():
            img, label, img_meta = item
            img = img.cuda()
            anno_items = img_meta['anno_item']
            img_names = img_meta['img_name']
            input_batch = {
            'img': img,
            "blood": img_meta['blood'].cuda(),
            'others': img_meta['others'].cuda(),
            "blood_des": clip.tokenize(img_meta['blood_des'], context_length=256).cuda(),
            "blood_des_1": clip.tokenize(img_meta['blood_des_1']).cuda(),
            "blood_des_2": clip.tokenize(img_meta['blood_des_2']).cuda(),
            "others_des": clip.tokenize(img_meta['others_des']).cuda(),
        }
            output = model(input_batch)
            preds_invade = output['preds_invade']
            preds_surgery = output['preds_surgery']
            preds_essential = output['preds_essential']
            preds_seg = output['pred_seg']
            feat = output['feat'].detach().cpu().numpy()
            # Fusion
            if test_fusion =='mean':
                pred_invade = torch.mean(torch.stack(preds_invade, dim=0), dim=0)
                pred_surgery = torch.mean(torch.stack(preds_surgery, dim=0), dim=0)
                pred_essential = torch.mean(torch.stack(preds_essential, dim=0), dim=0)
            if test_fusion == 'max':
                pred_invade,_ = torch.max(torch.stack(preds_invade, dim=0), dim=0)
                pred_surgery,_ = torch.max(torch.stack(preds_surgery, dim=0), dim=0)
                pred_essential,_ = torch.max(torch.stack(preds_essential, dim=0), dim=0)
                
            pred_invade = torch.softmax(pred_invade.detach().cpu(), dim=1)
            pred_surgery = torch.softmax(pred_surgery.detach().cpu(), dim=1)
            pred_essential = torch.softmax(pred_essential.detach().cpu(), dim=1)
            
            pred_invade_cls = torch.argmax(pred_invade,dim=1).numpy()
            pred_surgery_cls = torch.argmax(pred_surgery, dim=1).numpy()
            pred_essential_cls = torch.argmax(pred_essential, dim=1).numpy()
            
            for b in range(pred_invade.shape[0]):
                anno_item = anno_items[b]
                if anno_item not in list(pred_invade_dict.keys()):
                    pred_invade_dict[anno_item], pred_surgery_dict[anno_item], pred_essential_dict[anno_item] = {}, {}, {}
                    score_invade_dict[anno_item], score_surgery_dict[anno_item], score_essential_dict[anno_item] = {}, {}, {}
                    label_invade_dict[anno_item], label_surgery_dict[anno_item], label_essential_dict[anno_item] = {}, {}, {}
                    feat_dict[anno_item] = {}
                pred_invade_dict[anno_item][img_names[b]]= pred_invade_cls[b]
                pred_surgery_dict[anno_item][img_names[b]] = pred_surgery_cls[b]
                pred_essential_dict[anno_item][img_names[b]] = pred_essential_cls[b]
                score_invade_dict[anno_item][img_names[b]]. = pred_invade[b][1].item()
                score_surgery_dict[anno_item][img_names[b]] = pred_surgery[b][1].item()
                score_essential_dict[anno_item][img_names[b]] = pred_essential[b][1].item()
                label_invade_dict[anno_item][img_names[b]] = img_meta['label_invade'][b].item()
                label_surgery_dict[anno_item][img_names[b]] = img_meta['label_surgery'][b].item()
                label_essential_dict[anno_item][img_names[b]] = img_meta['label_essential'][b].item()
                feat_dict[anno_item][img_names[b]] = feat[b]
        save_dict = {
        'pred_invade_dict':pred_invade_dict,
        'pred_surgery_dict': pred_surgery_dict,
        "pred_essential_dict": pred_essential_dict,
        'score_invade_dict': score_invade_dict,
        "score_surgery_dict": score_surgery_dict,
        'score_essential_dict': score_essential_dict,
        'label_invade_dict': label_invade_dict,
        'label_surgery_dict': label_surgery_dict,
        'label_essential_dict': label_essential_dict,
        'feat_dict': feat_dict
    }
        return save_dict

# API端点，接收POST请求
@app.route('/segment', methods=['POST'])
def segment():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided."}), 400

        # 从请求中获取图像文件
        image_file = request.files['image']
        image = Image.open(image_file).convert('L')  # 转换为灰度图像
        image_tensor = preprocess_image(image)

        # 进行图像分割
        segmentation = segment_image(image_tensor)

        # 将分割结果图像转为Base64编码
        result_image = Image.fromarray(segmentation.astype('uint8'))
        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        result_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({"segmentation_image": result_image_base64})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/pancls',methods=['POST'])
def pancls():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided."}), 400

        # 从请求中获取图像文件
        image_file = request.files['image']
        image = Image.open(image_file).convert('L')  # 转换为灰度图像

        # 进行图像分割
        segmentation = segment_pancls_image(modelPancls,image_tensor,test_fusion="mean")

        # 将分割结果图像转为Base64编码
        # result_image = Image.fromarray(segmentation.astype('uint8'))
        # buffered = io.BytesIO()
        # result_image.save(buffered, format="PNG")
        # result_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify(segmentation)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
# 处理上传的图像并进行分割推理
