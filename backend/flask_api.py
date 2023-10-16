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
sys.path.append("/home/dmt218/zby/PANCLS")
from datasets.cellseg import CellSeg
from models.unetplusplus import NestedUNet
from utils.slide_infer import slide_inference
import models.text_encoder.clip as clip

sys.path.append(0,"/home/dmt218/zby/MTCSNet")
from datasets.WSCellseg import WSCellSeg
from models.unetplusplus import NestedUNet as NestedUNet2

app = Flask(__name__)
# 加载训练好的图像分割模型
modelPancls = NestedUNet()  # 定义了一个合适的模型类
modelPancls = torch.load("your_model_path.pth")  # 替换成你的模型路径
modelPancls.eval()
modelCellseg =NestedUNet2()
modelCellseg = torch.load("")
modelCellseg.eval()

pancls_dataset= CellSeg()
pancls_dataloader = DataLoader(pancls_dataset)
mtcs_dataset = WSCellSeg()
mtcs_dataloader= DataLoader(mtcs_dataset,batch_size= 1,num_workers=1)



# 图像预处理函数
def preprocess_pancls_image(image):
    
 
    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    image = transform(image).unsqueeze(0)
    return image

# 图像分割函数
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
        result_image = Image.fromarray(segmentation.astype('uint8'))
        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        result_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({"segmentation_image": result_image_base64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
# 处理上传的图像并进行分割推理
