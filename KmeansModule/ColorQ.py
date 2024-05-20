import cv2
import numpy as np
import math

#array의 값 생략 없이 출력
np.set_printoptions(threshold=np.inf, linewidth=np.inf)


img = cv2.imread('/Users/chaewookim/Desktop/ColorClassification/ColorDataSet/Red_1.jpg') #배열로 이미지 불러오기

#cv2.kmeans는 (N,3)의 shape과 flaot32의 데이터 형식을 입력 조건으로 사용 
#reshape(-1,3)은 2차원 배열로 바꾸되 3열로 바꾸고 행은 알아서 넣으라는 뜻
data = img.reshape((-1, 3)).astype(np.float32) #rgb 3차원 배열을 2차원으로 바꿔주는 것이 매우 중요

#criteria는 반복을 종료할 조건으로 (type, max_iter, epsilon) 총 3개의 인자 가짐
#type : 종료 조건의 타입. cv2.TERM_CRITERIA_EPS는 주어진 정확도인 epsilon에 도달하면 종료,
#cv2.TERM_CRITERIA_MAX_ITER는 max_iter에 지정된 횟수만큼 반복 후 종료. +로 연결해 사용하면 둘 중 하나 만족할 시 종료
#max_iter : 최대 반복 횟수, 정수형
#epsilon : 정확도
criteria = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER, 4, 1.0) #최대 10번 반복하고 정확도 1 이하로 떨어지면 종료.

#cv2.kmaens는 data, k, bestLabels, criteria, attempts, flags, centers를 인자로 갖고 retval, bestLabels, centers 리턴
#인자 중 besLabels는 None을 입력하고 결과값을 받음
#flags : 초기 중심값 위치에 대한 설정. cv2.KMEANS_RANDOM_CENTERS, cv2.KMEANS_PP_CENTERS, cv2.KMEANS_USE_INITIAL_LABELS 중 하나
#label은 해당 데이터가 어떤 군집에 속했는지를 알려주는 데이터인 0,1 등의 값을 갖고 있는 배열
#center은 군집의 중심점 좌표들이 저장된 배열 
#retval은 각 포인트와 중심 간의 거리의 제곱의 합. 높을수록 중심에서 멀리 있는 포인트들이 많고 낮을수록 중심에 가까이 있는 포인트들이 많음.
k=3
ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) #kmeans 실행 
#center는 float32 형식이므로 opencv에서 주로 사용하는 형식인 uint8로 변경 
center = np.uint8(center) 



#중심점 좌표를 받아서 dst에 입력 (307200, 3) 3은 중심 좌표
#flatten()은 다차원 배열을 1차원 배열로 평탄화 해주는 역할
#각 값들이 label에 저장되어 있는 값에 따라 중심의 색으로 치환됨
result = center[label.flatten()]
print(center)

#원본 사진과 동일한 shape으로 변환
convertedImage = result.reshape((img.shape))

blueImage = convertedImage.copy()
greenImage = convertedImage.copy()
redImage = convertedImage.copy()


blueImage[:,:,1] = 0 # G 제거
blueImage[:,:,2] = 0 # R 제거
greenImage[:,:,0] = 0 # B 제거
greenImage[:,:,2] = 0 # R 제거
redImage[:,:,0] = 0 # B 제거
redImage[:,:,1] = 0 # G 제거


height, width, channel = result.shape
bgr = convertedImage.astype(float) / 255.0

b, g, r = cv2.split(bgr)

h = np.zeros((height, width), dtype=float)
s = np.zeros((height, width), dtype=float)
v = np.max(bgr, axis=2)


for i in range(height):
    for j in range(width):
        if v[i][j] == 0:
            # print("v : ", v[i][j])
            h[i][j] = 0
            # print("h : ", h[i][j])
            s[i][j] = 0
            # print("s : ", s[i][j])
            # print(i*width + j, "번째 -> b : ", b[i][j], " g : ", g[i][j], " r : ", r[i][j])
            # print(i*width + j, "번째 -> h : ", h[i][j], " s : ", s[i][j], " v : ", v[i][j])
        else:
            min_rgb = min(bgr[i][j])
            # print("min_rgb : ", min_rgb)

            s[i][j] = 1 - (min_rgb / v[i][j])
            # print("s : ", s[i][j])

            if v[i][j] == r[i][j]:
                h[i][j] = 60 * (g[i][j] - b[i][j]) / (v[i][j] - min_rgb)
                # print("h : ", h[i][j])
            elif v[i][j] == g[i][j]:
                h[i][j] = 120 + (60 * (b[i][j] - r[i][j])) / (v[i][j] - min_rgb)
                # print("h : ", h[i][j])
            elif v[i][j] == b[i][j]:
                h[i][j] = 240 + (60 * (r[i][j] - g[i][j])) / (v[i][j] - min_rgb)
                # print("h : ", h[i][j])
            if h[i][j] < 0:
                h[i][j] += 360
                # print("h : ", h[i][j])
                h[i][j] /= 360
                # print("h : ", h[i][j])
            # print(i*width + j, "번째 -> b : ", b[i][j], " g : ", g[i][j], " r : ", r[i][j])
            # print(i*width + j, "번째 -> h : ", h[i][j], " s : ", s[i][j], " v : ", v[i][j])


            
hsv_img = (np.dstack((h, s, v)) * 255).astype(np.uint8)

f = open("h.txt", 'w+')
fstr = ''.join(str(h))
f.write(fstr)
f.close()
g = open("s.txt", 'w+')
gstr = ''.join(str(s*100))
g.write(gstr)
g.close()
a = open("v.txt", 'w+')
astr = ''.join(str(v*100))
a.write(astr)
a.close()
b = open("blue.txt", 'w+')
bstr = ''.join(str(blueImage))
b.write(bstr)
b.close()
c = open("green.txt", 'w+')
cstr = ''.join(str(greenImage))
c.write(cstr)
c.close()
d = open("red.txt", 'w+')
dstr = ''.join(str(redImage))
d.write(dstr)
d.close()





cv2.imshow('original', result)
cv2.waitKey()
# cv2.imshow('hsv', hsv_img)
# cv2.waitKey()
# cv2.imshow('h', h)
# cv2.waitKey()
# cv2.imshow('s', s)
# cv2.waitKey()
# cv2.imshow('v', v)
# cv2.waitKey()
