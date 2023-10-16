import os
import base64
import io
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify

from xxx import model
app = Flask(__name__)
# 加载训练好的图像分割模型
model = model()  # 定义了一个合适的模型类
model = torch.load("your_model_path.pth")  # 替换成你的模型路径
model.eval()

# 图像预处理函数
def preprocess_image(image):
    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    image = transform(image).unsqueeze(0)
    return image

# 图像分割函数
def segment_image(image):
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        return predicted[0].numpy()

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
# 处理上传的图像并进行分割推理
