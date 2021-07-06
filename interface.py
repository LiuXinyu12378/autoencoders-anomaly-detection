from resource.model import AnomalyAE
from PIL import Image
import torch
from torchvision.transforms import Compose, Grayscale, ToTensor
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

model = AnomalyAE()
model.load_state_dict(torch.load('./model/best_model_25_loss=-2.150636353559904e-06.pth',map_location=torch.device("cpu")))
model.eval()
model = model.to('cpu')
# imgpath = "./test_image/data2/0002.PNG"    #0.0579       #0.00479
# imgpath = "./test_image/data/0045.PNG"    #0.0008       #0.00141
# imgpath = "./Class1/Test/0001.PNG"       #0.134
imgpath = "./image/0045.PNG"       #0.006
img_yuan = cv2.imread(imgpath)
img = Image.open(imgpath).convert('L')
transform = Compose([Grayscale(), ToTensor()])
img = transform(img)
img = img.to('cpu')
img = img.unsqueeze(0)
y = model(img)
residual = torch.abs(img[0][0]-y[0][0])

# plt.figure(figsize=(15,10))
# plt.subplot(121),plt.imshow(img.detach().cpu().numpy()[0][0]),plt.title('Image'),plt.axis('off')
# print(img.detach().cpu().numpy()[0][0])
# plt.subplot(122),plt.imshow(residual.detach().cpu().numpy()>0.007),plt.title('Residual Thresholded'),plt.axis('off')

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
    #面积
    s = cv2.contourArea(contour)
    areas.append(s)

draw_img = img_yuan.copy()
if areas:
    all_area = res.shape[0]*res.shape[1]
    print("max_area_audio:",max(areas)/all_area)

    areas_idxs = []
    contour_areas = []
    for idx,area in enumerate(areas):
        if area/all_area > 0.001:
            areas_idxs.append(idx)
            contour_areas.append(contours[idx])
            x, y, w, h = cv2.boundingRect(contours[idx])
            img = cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 0, 255), 1)

#-1代表全画上，其他数值是框的索引

# res = cv2.drawContours(draw_img,contour_areas,-1,(0,255,0),1)

cv2.imshow("img_yuan",img_yuan)
cv2.imwrite("./image/img5.jpg",img_yuan)
cv2.imshow("bin_img",bin_img)
cv2.imwrite("./image/img6.jpg",bin_img)
cv2.imshow("result",result)
cv2.imwrite("./image/img7.jpg",result)
cv2.imshow("img",draw_img)
cv2.imwrite("./image/img8.jpg",draw_img)



# img1 = np.concatenate((img_yuan,bin_img,result,draw_img),axis=1)
# img2 = np.concatenate((bin_img,result),axis=1)
# cv2.namedWindow("img1",cv2.WINDOW_NORMAL)
# cv2.imshow("img1",img1)
# cv2.imshow("img2",img2)

cv2.waitKey(0)

# plt.savefig('sample_detection.png', bbox_inches='tight')

# plt.hist(residual.detach().cpu().numpy().ravel())
