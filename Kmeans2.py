from PIL import Image
import numpy as np
import pandas as pd
import random
import math
import sys
np.set_printoptions(threshold=sys.maxsize)


#해당 이미지를 배열로 변환
img = Image.open("pill.jpg")
pix = np.array(img)


#3차원 배열 pix를 2차원 배열로 변환 뒤 floor 열 삭제
row,col,floor = pix.shape
twoDim_array = np.column_stack(((np.repeat(np.arange(row), col)), pix.reshape(row*col, -1)))
twoDim_array = np.delete(twoDim_array, 0, axis=1)


class K_Means:
    #k와 학습시킬 data 받아오기
    #__init__은 반드시 첫 매개변수를 self로 해야함
    #객체 생성 시 이루어지는 것들 
    # : 서로 다른 값을 갖는 k개의 중심점 무작위로 선정되고 그 값이 standard에 저장
    # : 군집 저장할 Cluster의 요소 선정 및 구별
    #__intit__함수 까지는 정상 작동. 객체 만들어도 오류 나타나지 않음
    def __init__(self,k,data,row,col): #초기화 함수, 객체 생성 시 반드시 처음 호출.
        self.k = k
        self.data = data
        self.N = len(data) #데이터의 길이, 즉 데이터의 개수
        self.standard = [0 for _ in range(self.k)] #중심점의 값 저장
        self.distanceList = [0 for _ in range(self.k)] #데이터들의 각 중심점으로부터의 거리를 저장하는 리스트
        self.Cluster = [0 for _ in range(self.N)] #각 데이터들이 속해있는 군집의 중심점 값을 저장하는 리스트
        self.resultCluster = np.array([0 for _ in range(self.N)]) #색상 양자화를 진행 후 화면 출력에 사용할 배열
        self.hValue = np.zeros((row, col), dtype = float)
        self.sValue = np.zeros((row, col), dtype = float)
        self.vValue = np.zeros((row, col), dtype = float)


        #랜덤으로 초기 중심점 k개 설정
        tmp = random.sample(range(0,col*row),self.k)
        for i in range(self.k):
            self.standard[i] = self.data[tmp[i]]
        #0번째부터 시작하기 위해 row, col 모두 1씩 감소


    #클러스터링 하는 함수
    def clustering(self,clusteringCount): 
        for _ in range(clusteringCount): 
            #군집화 과정
            #1. 중심점과 각 점 사이의 거리 체크해서 가장 가까운 중심점에 속하게 함
            for i in range(self.N):
                for j in range(self.k):
                    self.distanceList[j] = (self.data[i][0]-self.standard[j][0])**2 + (self.data[i][1]-self.standard[j][1])**2 + (self.data[i][2]-self.standard[j][2])**2
                self.Cluster[i] = self.distanceList.index(min(self.distanceList))
            #중심점 재설정
            #1. 군집의 모든 BGR 값의 평균으로 중심점 설정
            for i in range(self.k):
                bCount, gCount, rCount = 1, 1, 1
                bSum, gSum, rSum = 0, 0, 0
                for j in range(self.N):
                    if (self.Cluster[j] == i):
                        bSum += self.data[j][0]
                        bCount += 1
                        gSum += self.data[j][1]
                        gCount += 1
                        rSum += self.data[j][2]
                        rCount += 1
                self.standard[i] = [int(bSum/bCount), int(gSum/gCount), int(rSum/rCount)]
            
        
    def quantization(self):
        #색상 양자화
        #데이터의 값을 가장 가까운 중심점의 값으로 변경
        for i in range(self.N):
            for j in range(self.k):
                if (self.Cluster[i] == j):
                    self.data[i] = self.standard[j]
        self.resultCluster = self.data.reshape((pix.shape))

    def bgrToHsv(self):
        #bgr을 hsv로 변환
        for i in range(self.N):
            maximum, minimum = 0, 0
            b = float(self.data[i][0])/255
            g = float(self.data[i][1])/255
            r = float(self.data[i][2])/255
            maximum = max(b,g,r)
            minimum = min(b,g,r)

            v = maximum

            if (v == 0):
                h = 0
                s = 0
            else:
                s = 1-(minimum/v)
                if (v == r):
                    h = 60*(g-b)/(v-minimum)
                elif (v == g):
                    h = 120 + (60*(b-r))/(v-minimum)
                elif (v == b):
                    h = 240 + (60*(r-g))/(v-minimum)
                if (h < 0):
                    h += 360
                    h /= 360

            input_row = int(i/(col))
            input_col = int(i%(col))
            self.hValue[input_row][input_col] = h
            self.sValue[input_row][input_col] = s
            self.vValue[input_row][input_col] = v
        
            
    def valueToText(self):
        self.blueImage = self.resultCluster.copy()
        self.greenImage = self.resultCluster.copy()
        self.redImage = self.resultCluster.copy()
        self.blueImage[:,:,1] = 0
        self.blueImage[:,:,2] = 0
        self.greenImage[:,:,0] = 0
        self.greenImage[:,:,2] = 0
        self.redImage[:,:,0] = 0
        self.redImage[:,:,1] = 0
        hValueText = open("h.txt", 'w+')
        hToString = ''.join(str(self.hValue))
        hValueText.write(hToString)
        hValueText.close()
        sValueText = open("s.txt", 'w+')
        sToString = ''.join(str(self.sValue*100))
        sValueText.write(sToString)
        sValueText.close()
        vValueText = open("v.txt", 'w+')
        vToString = ''.join(str(self.vValue*100))
        vValueText.write(vToString)
        vValueText.close()
        blueValueText = open("blue.txt", 'w+')
        blueToString = ''.join(str(self.blueImage))
        blueValueText.write(blueToString)
        greenValueText = open("green.txt", 'w+')
        greenToString = ''.join(str(self.greenImage))
        greenValueText.write(greenToString)
        redValueText = open("red.txt", 'w+')
        redToString = ''.join(str(self.redImage))
        redValueText.write(redToString)    


    def show(self):
        self.resultCluster = self.data.reshape((pix.shape))
        resultImage = Image.fromarray(self.resultCluster.astype(np.uint8))
        hImage = Image.fromarray(self.hValue.astype(np.uint8))
        sImage = Image.fromarray(self.sValue.astype(np.uint8))
        vImage = Image.fromarray(self.vValue.astype(np.uint8))
        hsvImage = Image.fromarray((np.dstack((self.hValue,self.sValue,self.vValue)) * 255).astype(np.uint8))
        hsvImage.show()
        hImage.show()
        sImage.show()
        vImage.show()
        resultImage.show()
        
        
Algorithm = K_Means(k=3, data=twoDim_array, row=row, col=col)
Algorithm.clustering(4)
Algorithm.quantization()
Algorithm.bgrToHsv()
Algorithm.valueToText()
Algorithm.show()
