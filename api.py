import requests
import json
import base64
import os
import numpy as np
import math
import cv2
import sys
import time

API_CATEGORY = os.environ.get("API_CATEGORY", "cv")
API_NAME = os.environ.get("API_NAME", "abnormal_texture")
API_VERSION = os.environ.get("API_VERSION", "1.0")
DEUBG_MODE = os.environ.get("DEBUG_MODE", "True")

API_URI = '/%s/%s/%s' % (API_CATEGORY, API_NAME, API_VERSION)
Healthy_URI = '%s/%s' % (API_URI, "healthy")

# print(API_URI)

def pic2base64(image_path):

    with open(image_path, 'rb') as f:
        image = f.read()
    image_base64 = base64.b64encode(image)
    image_base64 = image_base64.decode()

    return image_base64


def post_request(imageId,image_path,url=None):

    url_ = 'http://10.110.156.101:8082'+API_URI
    base64Data = pic2base64(image_path)
    format = os.path.splitext(image_path)[-1].replace(".","")
    url = ""

    data = {"imageId": imageId,
            "base64Data": base64Data,
            "format": format,
            "url": url}

    data = json.dumps(data)
    # print(data)
    res = requests.post(url_, data=data).text
    result = json.loads(res)["result"]
    print(res,type(res))

    if result["target"] == "yes":
        img = cv2.imread(image_path)
        draw_img = img.copy()
        for x,y,w,h in result["delects"]:
            cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.imshow("draw_img",draw_img)
        cv2.waitKey(0)

if __name__ == '__main__':
    post_request("00001","image/0002.png")
