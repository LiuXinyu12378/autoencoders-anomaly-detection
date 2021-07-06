from .resource.model import AnomalyAE
from PIL import Image
import torch
from torchvision.transforms import Compose, Grayscale, ToTensor
import matplotlib.pyplot as plt
import base64
import json
import cv2
import numpy as np
from flask import Flask,request,jsonify
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

base_path = os.path.dirname(__file__)

CUDA = False
app = Flask(__name__)
API_CATEGORY = os.environ.get("API_CATEGORY", "cv")
API_NAME = os.environ.get("API_NAME", "abnormal_texture")
API_VERSION = os.environ.get("API_VERSION", "1.0")
DEUBG_MODE = os.environ.get("DEBUG_MODE", "True")

API_URI = '/%s/%s/%s' % (API_CATEGORY, API_NAME, API_VERSION)
Healthy_URI = '%s/%s' % (API_URI, "healthy")

if CUDA ==True:
    device = "cuda"
else:
    device = "cpu"

model = AnomalyAE()
model.load_state_dict(torch.load('./model/best_model_25_loss=-2.150636353559904e-06.pth',map_location=torch.device(device)))
model.eval()
model = model.to(device)

def base64_2_pic(base64data):
    base64data = base64data.encode(encoding="utf-8")
    data = base64.b64decode(base64data)
    imgstring = np.array(data).tostring()
    imgstring = np.asarray(bytearray(imgstring), dtype="uint8")
    image = cv2.imdecode(imgstring, cv2.IMREAD_COLOR)

    return image


@app.route(API_URI, methods=["POST"])
def interface():
    data = request.get_json()
    if data:
        pass
    else:
        data = request.get_data()
        data = json.loads(data)

    imageId = data["imageId"]
    base64Data = data["base64Data"]
    format = data["format"]
    url = data["url"]

    image = base64_2_pic(base64Data)

    image = cv2.resize(image,(512,512))

    result = delect(image)
    data = {
        "status": 0,
        "message": "success",
        "result":result
    }
    return data




def delect(image):
    image2 = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    transform = Compose([Grayscale(), ToTensor()])
    img = transform(image2)
    img = img.to(device)
    img = img.unsqueeze(0)
    y = model(img)
    residual = torch.abs(img[0][0]-y[0][0])

    res = residual.detach().cpu().numpy()>0.007
    res = np.uint8(res)

    #统计二值化中1出现的次数
    cout = np.sum(res)
    all = res.shape[0]*res.shape[1]
    area_ratio = cout/all    #1的比率
    print(area_ratio)

    ret,bin_img = cv2.threshold(res,0,255,cv2.THRESH_BINARY)

    if area_ratio>0.005:    #认为可能是平滑的纹理缺陷
        kernel = np.ones((7, 7), np.uint8)
        dilate = cv2.dilate(bin_img, kernel, iterations=2)
        bit_fanzhuan = cv2.bitwise_not(dilate,dilate)
        kernel = np.ones((7,7),np.uint8)
        opening = cv2.morphologyEx(bit_fanzhuan,cv2.MORPH_OPEN,kernel)
        result = opening

    else:
        kernel = np.ones((7,7),np.uint8)
        closing = cv2.morphologyEx(bin_img,cv2.MORPH_CLOSE,kernel)
        kernel = np.ones((7,7),np.uint8)
        opening = cv2.morphologyEx(closing,cv2.MORPH_OPEN,kernel)
        # median = cv2.medianBlur(closing, 7)
        result = opening

    binary,contours,hierarchy = cv2.findContours(result,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    areas = []
    for contour in contours:
        s = cv2.contourArea(contour)  #计算面积
        areas.append(s)

    target = "no"

    delects = []
    if areas:
        all_area = res.shape[0]*res.shape[1]
        print("max_area_audio:",max(areas)/all_area)
        for idx,area in enumerate(areas):
            if area/all_area > 0.001:
                x, y, w, h = cv2.boundingRect(contours[idx])
                delects.append([x,y,w,h])
    if delects:
        target = "yes"
    else:
        target = "no"

    data = {
        "delects":delects,
        "target":target
    }
    return data




if __name__ == '__main__':
    app.run("0.0.0.0",port=8080)