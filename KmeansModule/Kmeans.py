#최종 수정일 : 2024.05.21
#오류 : 엑셀에 저장될 때 가끔 Final Color가 결정되지 않는 오류 발생
#수정 : colorDet
#해결 : 지나치게 복잡한 코드를 두고 새로 작성

from PIL import Image
import pandas as pd
import numpy as np
import random
import sys
import math
import openpyxl
import os
import csv

import clustering


colorList = ["Red", "Yellow", "Green", "Blue", "Black", "Grey", "White"]
colorAccuracyList = [0 for _ in range(7)]
imgAllCount = [0 for _ in range(7)]
correctImgCount = [0 for _ in range(7)]

for z in range(7):
    # if z == 0 or z == 1 or z == 4 or z == 6 or z == 5:
    #     continue
    if z == 0:
        imgAmount = 200 # Red 이미지 개수
    elif z == 1:
        imgAmount = 200 # Yellow 이미지 개수
    elif z == 2:
        imgAmount = 200 # Green 이미지 개수
    elif z == 3:
        imgAmount = 199 # Blue 이미지 개수
    elif z == 4:
        imgAmount = 200 # Black 이미지 개수
    elif z == 5:
        imgAmount = 100 # Grey 이미지 개수
    elif z == 6:
        imgAmount = 118 # White 이미지 개수

    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        
    # 색상 변경 시 바꿀 부분
    
    imgEnd = 0
    imgArr = ["" for _ in range(imgAmount)]
    imgAddress = "/Users/gimchaeu/Desktop/알약 과제/pngImg/"

    imgObject = " Car"
    imgColor = colorList[z]
    k=5
    imgType = " png"
    imgSlash = "/"
    imgPng = ".png"
    fileName = '2024_05_21.xlsx' # 저장에 사용할 엑셀 파일 이름


    correctCount = 0 # 이미지 중 imgColor와 finalColor가 같은 이미지의 개수
    colorAccuracy = 0 # 모든 이미지에 대한 정확도

    searchKey = imgColor + imgObject + imgType # 검색어


    # 파일에서 여러 이미지를 가져오기 때문에 더 이상 사용하지 않음
    # 해당 이미지를 배열로 변환
    # img = Image.open("/Users/gimchaeu/Desktop/ColorClassification/red/1.jpg")
    # pix = np.array(img)
    # k=7
    # imageId = 'White_5'
    # sheetName = 'White Image'

    # #3차원 배열 pix를 2차원 배열로 변환 뒤 floor 열 삭제
    # row,col,floor = pix.shape
    # twoDim_array = np.column_stack(((np.repeat(np.arange(row), col)), pix.reshape(row*col, -1)))
    # twoDim_array = np.delete(twoDim_array, 0, axis=1)

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
            self.standardIndex = [0 for _ in range(self.k)] #중심점의 좌표 저장
            # print(self.standardIndex)
            self.distanceList = [0 for _ in range(self.k)] #데이터들의 각 중심점으로부터의 거리를 저장하는 리스트
            self.Cluster = [0 for _ in range(self.N)] #각 데이터들이 속해있는 군집의 중심점 값을 저장하는 리스트
            self.resultCluster = np.array([0 for _ in range(self.N)]) #색상 양자화를 진행 후 화면 출력에 사용할 배열
            self.hValue = np.zeros((row, col), dtype = float)
            self.sValue = np.zeros((row, col), dtype = float)
            self.vValue = np.zeros((row, col), dtype = float)
            self.bgr = [[0,0,0] for _ in range(self.k)]
            self.hsv = [[0,0,0] for _ in range(self.k)]
            # print(self.N, col, row)
            self.eachCount = [0 for _ in range(self.k)]

            self.backgroundColor = [0 for _ in range(self.k-1)]

            #Color
            self.finalColor = ""
            self.color = ["Color" for _ in range(self.N)] #세분화 되었을 때의 색
            self.colorCount = [0 for _ in range(7)] #각 색이 몇 개인지 확인하는 용도. 빨/주/노/연/녹/청/파/남/보/검/회/흰 순서
            self.colorList = ["Red", "Yellow", "Green", "Blue", "Black", "Grey", "White"]
            self.redCount = 0
            self.orangeCount = [0 for _ in range(6)]
            self.yellowCount = 0
            self.lightGreenCount = [0 for _ in range(5)]
            self.greenCount = 0
            self.cyanCount = [0 for _ in range(6)]
            self.blueCount = 0
            self.indigoCount = [0 for _ in range(2)]
            self.magentaCount = [0 for _ in range(4)]
            self.blackCount = 0
            self.greyCount = 0
            self.whiteCount = 0
            self.allColorCount = [self.redCount, self.yellowCount, self.greenCount, self.blueCount, self.blackCount, self.greyCount, self.whiteCount]
            #ColorType
            self.redColor = "Red"
            self.orangeColor = "Orange"
            self.yellowColor = "Yellow"
            self.lightGreenColor = ["LightGreen(G70Y)", "LightGreen(G60Y)", "LightGreen(G50Y)", "LightGreen(G40Y)", "LightGreen(G30Y)"]
            self.greenColor = "Green"
            self.cyanColor = ["Cyan(B70G)", "Cyan(B60G)", "Cyan(B50G)", "Cyan(B40G)", "Cyan(B30G)", "Cyan(B20G)"]
            self.blueColor = "Blue"
            self.indigoColor = ["Indigo(R70B)", "Indigo(R60B)"]
            self.magentaColor = ["Magenta(R50B)", "Magenta(R40B)", "Magenta(R30B)", "Magenta(R20B)"]
            self.blackColor = "Black"
            self.greyColor = "Grey"
            self.whiteColor = "White"
            self.allColor = ["Red", "Yellow", "Green", "Blue", "Black", "Grey", "White"]
            #StartColor
            self.redStartColor = [314, 0] 
            self.orangeStartColor = [12, 20, 25, 30, 33, 37]
            self.yellowStartColor = [32]
            self.lightGreenStartColor = [56, 61, 65, 68, 75]
            self.greenStartColor = [64]
            self.cyanStartColor = [171, 173, 176, 178, 181, 184]
            self.blueStartColor = [177]
            self.indigoStartColor = [209, 235]
            self.magentaStartColor = [260, 286, 314, 326]
            self.allStartColor = [self.redStartColor, self.yellowStartColor, self.greenStartColor, self.blueStartColor]
            #EndColor
            self.redEndColor = [360, 32]
            self.orangeEndColor = [19, 24, 29, 32, 36, 40]
            self.yellowEndColor = [64]
            self.lightGreenEndColor = [60, 64, 67, 74, 88]
            self.greenEndColor = [177]
            self.cyanEndColor = [172, 175, 177, 180, 183, 186]
            self.blueEndColor = [314]
            self.indigoEndColor = [234, 259]
            self.magentaEndColor = [285, 313, 325, 336]
            self.allEndColor = [self.redEndColor, self.yellowEndColor, self.greenEndColor, self.blueEndColor]
            self.rgbEx = [0 for _ in range(self.k)]
            self.hsvEx = [0 for _ in range(self.k)]
            self.largeColorEx = ["Color" for _ in range(self.k)]
            self.smallColorEx = ["Color" for _ in range(self.k)]
            self.pixNumEx = [0 for _ in range(self.k)]
            self.pixRatioEx = [0 for _ in range(self.k)]
            self.imageId = [imageId for _ in range(self.k)]
            self.pixNumSum = [col*row for _ in range(self.k)]
            self.kNum = [self.k for _ in range(self.k)]
            self.refColor = ["Color" for _ in range(self.k)]
            self.refColorCom = [0 for _ in range(len(self.colorCount))]


            # print("col : ", col)
            # print("row : ", row)
            #랜덤으로 초기 중심점 k개 설정
            tmp = random.sample(range(0,col*row),self.k)
            for i in range(self.k):
                self.standard[i] = self.data[tmp[i]]
                self.standardIndex[i] = tmp[i]
                # print(i,"번째 중심점 좌표 : [", self.standardIndex[i]/col, ", ", self.standardIndex[i]%col, "]")
            

        #클러스터링 하는 함수
        def clustering(self,clusteringCount): 
            for _ in range(clusteringCount): 
                #군집화 과정
                #1. 중심점과 각 점 사이의 거리 체크해서 가장 가까운 중심점에 속하게 함
                for i in range(self.N):
                    for j in range(self.k):
                        self.distanceList[j] = (self.data[i][0]-self.standard[j][0])**2 + (self.data[i][1]-self.standard[j][1])**2 + (self.data[i][2]-self.standard[j][2])**2
                        # self.distanceList[j] = (int(self.standardIndex[j])/col-i/col)**2 + (int(self.standardIndex[j])%col-i%col)**2
                    self.Cluster[i] = self.distanceList.index(min(self.distanceList))

                #2. 중심점의 좌표 값 재설정
                for i in range(self.k):
                    rowSum = 0.0
                    colSum = 0.0
                    rowCount = 1 #일단 1로 하고 나중에 수정할 것
                    colCount = 1 #일단 1로 하고 나중에 수정할 것
                    for j in range(self.N):
                        if (self.Cluster[j] == i):
                            rowSum = rowSum + j/col
                            colSum = colSum + j%col
                            rowCount = rowCount + 1
                            colCount = colCount + 1
                    self.standardIndex[i] = int(rowSum/rowCount*col) + int(colSum/colCount) #0으로 나눠지는 오류 발생. rowCount, colCount를 1로 하면 될듯?
                    self.standard[i] = self.data[self.standardIndex[i]]

                #중심점의 값 재설정
                #1. 군집의 모든 BGR 값의 평균으로 중심점 설정
                for i in range(self.k):
                    rCount, gCount, bCount = 1, 1, 1
                    rSum, gSum, bSum = 0, 0, 0
                    for j in range(self.N):
                        if (self.Cluster[j] == i):
                            rSum += self.data[j][0]
                            rCount += 1
                            gSum += self.data[j][1]
                            gCount += 1
                            bSum += self.data[j][2]
                            bCount += 1
                    self.standard[i] = [int(rSum/rCount), int(gSum/gCount), int(bSum/bCount)]
                
            
        def quantization(self):
            #색상 양자화
            #데이터의 값을 가장 가까운 중심점의 값으로 변경
            # print("standard: ",self.standard)
            # dataValueText = open("data.txt", 'w+')
            # dataToString = ''.join(str(self.data))
            # dataValueText.write(dataToString)
            # print(self.data.shape)

            #알파 채널 분리
            #1. 알파 채널의 값이 0이면 나머지 값을 모두 -1로, 0이 아닌 다른 수면 유지
            #2. 4열 삭제

            #1. 값 변경
            if (floor == 4):
                for i in range(self.N):
                    #0일 때, 즉 배경일 때
                    if (np.array_equal(self.data[i], [0,0,0,0])):
                        self.data[i] = [-1, -1, -1, -1]
                #2. 4열 삭제
                self.data = np.delete(self.data, 3, axis=1)
                # dataValueText = open("data.txt", 'w+')
                # dataToString = ''.join(str(self.data))
                # dataValueText.write(dataToString)
                

            for i in range(self.N):
                for j in range(self.k):
                    if (self.Cluster[i] == j):
                        if (np.array_equal(self.data[i], [-1, -1, -1])):
                            continue
                        self.data[i] = self.standard[j]
                        self.eachCount[j] = self.eachCount[j]+1
            # print("양자화 후 중심점 값 : ", self.standard)
            dataValueText = open("data.txt", 'w+')
            dataToString = ''.join(str(self.data))
            dataValueText.write(dataToString)


        def bgrToHsv(self):
            #bgr을 hsv로 변환
            for i in range(self.N):
                if (np.array_equal(self.data[i], [-1, -1, -1])):
                    self.hValue[int(i/(col))][int(i%(col))] = -1
                    self.sValue[int(i/(col))][int(i%(col))] = -1
                    self.vValue[int(i/(col))][int(i%(col))] = -1
                    continue

                maximum, minimum = 0, 0

                r = float(self.data[i][0])/255
                g = float(self.data[i][1])/255
                b = float(self.data[i][2])/255
                maximum = max(r,g,b)
                minimum = float(min(r,g,b)) - 0.000001

                v = maximum
                h = 0
                s = 0

                if ((r == g) and (g == b)):
                    v = v + 1/255

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
                self.hValue[input_row][input_col] = round(h)
                self.sValue[input_row][input_col] = round(s*100)
                self.vValue[input_row][input_col] = round(v*100)


        def bgrToHsvOne(self, input):
            #bgr을 hsv로 변환
            maximum, minimum = 0.0, 0.0
            r = float(input[0])/255
            g = float(input[1])/255
            b = float(input[2])/255
            maximum = float(max(r,g,b))
            minimum = float(min(r,g,b)) - 0.000001

            v = maximum
            h = 0.0
            s = 0.0

            print("r : ", r, " g : ", g, " b : ", b, " max : ", maximum, " min : ", minimum, " h : ", h, " s : ", s, " v : ", v)

            # if ((r == g) and (g == b)):
            #     v = v + 1/255

            if (v == 0.0):
                h = 0.0
                s = 0.0
            else:
                s = 1-(minimum/v)
                if (v == r):
                    # print("h, s, v, min, max: ", h, s, v, minimum, maximum)
                    h = 60*(g-b)/(v-minimum)
                elif (v == g):
                    h = 120 + (60*(b-r))/(v-minimum)
                elif (v == b):
                    h = 240 + (60*(r-g))/(v-minimum)
                if (h < 0):
                    h += 360
                    h /= 360
                    
            return [round(h), round(s*100), round(v*100)]


        def colorInput(self, index, colorNumber):
            #1. 색 소분류 결과 값 저장
            #2. 색 대분류 카운트 증가
            #3. 색 소분류 카운트 증가
            #검회흰색을 넣고 이중으로 카운트 하지 않기 위한 조건문 추가
            if (self.color[index] == "Color"):
                self.color[index] = self.allColor[colorNumber]
                self.colorCount[colorNumber] = self.colorCount[colorNumber] + 1
                self.allColorCount[colorNumber] = self.allColorCount[colorNumber] + 1
            return self.allColor[colorNumber]


        #색 소분류
        def colorSmallClassify(self, index, colorNumber):
            if (self.hValue[int(index/col)][int(index%col)] <= 32 and self.hValue[int(index/col)][int(index%col)] > 314):
                Algorithm.colorInput(index, colorNumber)
            elif (self.hValue[int(index/col)][int(index%col)] > 32 and self.hValue[int(index/col)][int(index%col)] <= 64):
                Algorithm.colorInput(index, colorNumber)
            elif (self.hValue[int(index/col)][int(index%col)] > 64 and self.hValue[int(index/col)][int(index%col)] <= 177):
                Algorithm.colorInput(index, colorNumber)
            elif (self.hValue[int(index/col)][int(index%col)] > 177 and self.hValue[int(index/col)][int(index%col)] <= 314):
                Algorithm.colorInput(index, colorNumber)
            else:
                Algorithm.colorInput(index, colorNumber)

        #색 대분류
        def colorLargeClassify(self, index):
            if (self.hValue[int(index/col)][int(index%col)] <= 32 and self.hValue[int(index/col)][int(index%col)] > 314):
                Algorithm.colorSmallClassify(index, 0)
            elif (self.hValue[int(index/col)][int(index%col)] > 32 and self.hValue[int(index/col)][int(index%col)] <= 64):
                Algorithm.colorSmallClassify(index, 1)
            elif (self.hValue[int(index/col)][int(index%col)] > 64 and self.hValue[int(index/col)][int(index%col)] <= 177):
                Algorithm.colorSmallClassify(index, 2)
            elif (self.hValue[int(index/col)][int(index%col)] > 177 and self.hValue[int(index/col)][int(index%col)] <= 314):
                Algorithm.colorSmallClassify(index, 3)
            else:
                Algorithm.colorSmallClassify(index, 0)


        def colorFiguration(self):
            #대분류, 소분류 함수들 호출
            for index in range(self.N):
                if (self.data[index][0] == -1):
                    continue
                #Black
                if ((self.sValue[int(index/col)][int(index%col)] < 20) and (self.vValue[int(index/col)][int(index%col)] <= 20) and (self.vValue[int(index/col)][int(index%col)] >= 0)):
                    Algorithm.colorInput(index, 4)

                #Grey       
                if ((self.sValue[int(index/col)][int(index%col)] >= 0) and (self.sValue[int(index/col)][int(index%col)] < 4) and (self.vValue[int(index/col)][int(index%col)] > 20) and (self.vValue[int(index/col)][int(index%col)] <= 90)):
                    Algorithm.colorInput(index, 5)
                if ((self.sValue[int(index/col)][int(index%col)] >= 4) and (self.sValue[int(index/col)][int(index%col)] < 8) and (self.vValue[int(index/col)][int(index%col)] > 20) and (self.vValue[int(index/col)][int(index%col)] <= 70)):
                    Algorithm.colorInput(index, 5)
                if ((self.sValue[int(index/col)][int(index%col)] >= 8) and (self.sValue[int(index/col)][int(index%col)] < 12) and (self.vValue[int(index/col)][int(index%col)] > 20) and (self.vValue[int(index/col)][int(index%col)] <= 50)):
                    Algorithm.colorInput(index, 5)
                if ((self.sValue[int(index/col)][int(index%col)] >= 12) and (self.sValue[int(index/col)][int(index%col)] < 16) and (self.vValue[int(index/col)][int(index%col)] > 20) and (self.vValue[int(index/col)][int(index%col)] <= 30)):
                    Algorithm.colorInput(index, 5)

                #White
                if ((self.sValue[int(index/col)][int(index%col)] >= 0) and (self.sValue[int(index/col)][int(index%col)] < 4) and (self.vValue[int(index/col)][int(index%col)] > 90) and (self.vValue[int(index/col)][int(index%col)] <= 100)):
                    Algorithm.colorInput(index, 6)

                #나머지 색들은 s,v를 고려하지 않기 때문에 검흰회색보다 나중에 실행
                Algorithm.colorLargeClassify(index)

        
        def colorFigurationForExcel(self, index):
            #엑셀 저장용 색상 대분류 위한 함수
            #Black
            if ((self.sValue[int(index/col)][int(index%col)] < 20) and (self.vValue[int(index/col)][int(index%col)] <= 20) and (self.vValue[int(index/col)][int(index%col)] >= 0)):
                return "Black"

            #Grey       
            if ((self.sValue[int(index/col)][int(index%col)] >= 0) and (self.sValue[int(index/col)][int(index%col)] < 4) and (self.vValue[int(index/col)][int(index%col)] > 20) and (self.vValue[int(index/col)][int(index%col)] <= 90)):
                return "Grey"
            if ((self.sValue[int(index/col)][int(index%col)] >= 4) and (self.sValue[int(index/col)][int(index%col)] < 8) and (self.vValue[int(index/col)][int(index%col)] > 20) and (self.vValue[int(index/col)][int(index%col)] <= 70)):
                return "Grey"
            if ((self.sValue[int(index/col)][int(index%col)] >= 8) and (self.sValue[int(index/col)][int(index%col)] < 12) and (self.vValue[int(index/col)][int(index%col)] > 20) and (self.vValue[int(index/col)][int(index%col)] <= 50)):
                return "Grey"
            if ((self.sValue[int(index/col)][int(index%col)] >= 12) and (self.sValue[int(index/col)][int(index%col)] < 16) and (self.vValue[int(index/col)][int(index%col)] > 20) and (self.vValue[int(index/col)][int(index%col)] <= 30)):
                return "Grey"

            #White
            if ((self.sValue[int(index/col)][int(index%col)] >= 0) and (self.sValue[int(index/col)][int(index%col)] < 4) and (self.vValue[int(index/col)][int(index%col)] > 90) and (self.vValue[int(index/col)][int(index%col)] <= 100)):
                return "White"
            
            #나머지 색들은 s,v를 고려하지 않기 때문에 검흰회색보다 나중에 실행
            return Algorithm.colorLargeClassifyForExcel(index)


        #색 대분류
        def colorLargeClassifyForExcel(self, index):
            if (self.hValue[int(index/col)][int(index%col)] <= 32 or self.hValue[int(index/col)][int(index%col)] > 314):
                return "Red"
            elif (self.hValue[int(index/col)][int(index%col)] > 32 and self.hValue[int(index/col)][int(index%col)] <= 64):
                return "Yellow"
            elif (self.hValue[int(index/col)][int(index%col)] > 64 and self.hValue[int(index/col)][int(index%col)] <= 177):
                return "Green"
            else:
                return "Blue"
                        

        def colorFigurationForExcelSmall(self, index):
            #엑셀 저장용 색상 대분류 위한 함수
            #Black
            if ((self.sValue[int(index/col)][int(index%col)] < 20) and (self.vValue[int(index/col)][int(index%col)] <= 20) and (self.vValue[int(index/col)][int(index%col)] >= 0)):
                return "Black"

            #Grey       
            if ((self.sValue[int(index/col)][int(index%col)] >= 0) and (self.sValue[int(index/col)][int(index%col)] < 4) and (self.vValue[int(index/col)][int(index%col)] > 20) and (self.vValue[int(index/col)][int(index%col)] <= 90)):
                return "Grey"
            if ((self.sValue[int(index/col)][int(index%col)] >= 4) and (self.sValue[int(index/col)][int(index%col)] < 8) and (self.vValue[int(index/col)][int(index%col)] > 20) and (self.vValue[int(index/col)][int(index%col)] <= 70)):
                return "Grey"
            if ((self.sValue[int(index/col)][int(index%col)] >= 8) and (self.sValue[int(index/col)][int(index%col)] < 12) and (self.vValue[int(index/col)][int(index%col)] > 20) and (self.vValue[int(index/col)][int(index%col)] <= 50)):
                return "Grey"
            if ((self.sValue[int(index/col)][int(index%col)] >= 12) and (self.sValue[int(index/col)][int(index%col)] < 16) and (self.vValue[int(index/col)][int(index%col)] > 20) and (self.vValue[int(index/col)][int(index%col)] <= 30)):
                return "Grey"

            #White
            if ((self.sValue[int(index/col)][int(index%col)] >= 0) and (self.sValue[int(index/col)][int(index%col)] < 4) and (self.vValue[int(index/col)][int(index%col)] > 90) and (self.vValue[int(index/col)][int(index%col)] <= 100)):
                return "White"
            
            #나머지 색들은 s,v를 고려하지 않기 때문에 검흰회색보다 나중에 실행
            return Algorithm.colorLargeClassify(index)


        def colorDet(self):
            #con1: Pixel Ratio가 가장 높은 색상
            #con2: Big Color 수가 많은 색상

            #기준
            #1. con1 && con2 -> 대표색 결정
            #2. con2 우선
            #3. con1일 때 -> Big Color이 판별될 것
            #4. 그 중 Pixel Ratio가 가장 높은 Small Color를 이미지의 대표색으로 선정
            #- 어디에도 걸리지 않는 경우 etc로 설정

            #con1: self.pixNumEx에서 가장 높은 값을 갖는 인덱스 가져옴 -> self.largeColorEx로 색상 가져와 저장
            #con2: Big Color의 개수 확인

            highPixIndex = 0
            highPixNum = 0

            bigColorName = ["" for _ in range(self.k)] #Big Color의 이름
            bigColorNum = [0 for _ in range(self.k)] #Big Color의 개수

            isCon1 = False
            isCon2 = False
            con1Color = ""
            con2Color = ""

            #con1
            for i in range(len(self.pixNumEx)):
                if (self.pixNumEx[i] >= highPixNum):
                    if (self.rgbEx[i] == "배경"):
                        print("배경: ", self.rgbEx[i], " : ", i)
                        continue
                    highPixNum = self.pixNumEx[i]
                    highPixIndex = i
                    isCon1 = True
                    print(i, "번째: ", highPixIndex)
                    print(i, "번째: ", self.rgbEx[i])
                    # break
            con1Color = self.largeColorEx[highPixIndex]
            print("con1: ", isCon1, " / con1Color: ",con1Color)

            print(self.largeColorEx)
            print(bigColorName)
            print(bigColorNum)

            #con2
            #self.largeColorEx에서 색상 개수만 세면 됨
            compare = ""

            #bigColorName과 bigColorNum에 Big Color 색상의 이름과 각 색상 별 개수 넣기
            for i in range(self.k):
                if compare == "배경":
                    #배경이면 bigColorName은 배경, bigColrNum은 초기값 0인 그대로
                    bigColorName[i] = "배경"
                    continue
                if compare not in bigColorName[i]:
                    #compare의 값이 bigColorName에 저장되어 있는 값이 아닐 때, 즉 처음 들어오는 big Color일 때
                    bigColorNum[i] += 1
                    bigColorName[i] = self.largeColorEx[i]
                    compare = self.largeColorEx[i]
                else:
                    bigColorNum[bigColorName.index(compare)] += 1

                    #i가 마지막 순서일 때 끝내지 않으면 아래 블럭에서 오류 발생
                    #self.largeColorEx는 마지막 번지가 k-1이니까
                    if i == self.k-1:
                        break

                    #다음에 비교할 값 넣어주기
                    compare = self.largeColorEx[i+1]

            count = 0

            #con2에 맞는 색상 결정
            #가장 큰 값 결정
            for i in range(self.k):
                if bigColorName[i] == "배경":
                    continue
                if bigColorNum[i] >= count:
                    count = bigColorNum[i]
                    con2Color = bigColorName[i]
                    isCon2 = True

            #con1, con2 종합해 final Color 결정
            if isCon1 and isCon2:
                self.finalColor = con1Color
            elif isCon2 == True:
                self.finalColor = con2Color
            elif isCon1 == True:
                self.finalColor = con1Color
            else:
                self.finalColor = "etc"
                
            #아래 코드는 색이 더 추가될 때 사용하거나 아예 사용하지 않을 예정. 위 코드보다 복잡하고 효율성 떨어짐
            # index = 0

            # for i in range(len(self.largeColorEx)):
            #     if (self.rgbEx[i] == "배경"):
            #             continue
            #     if self.largeColorEx[i] in bigColorName:
            #         bigColorNum[bigColorName.index(self.largeColorEx[i])] += 1
            #     else:
            #         bigColorName[i] = self.largeColorEx[i]
            #         bigColorNum[bigColorName.index(self.largeColorEx[i])] += 1
            
            # for i in range(len(bigColorNum)):
            #     if (self.rgbEx[i] == "배경"):
            #             continue
            #     if (bigColorNum[i] > maxCount):
            #         index = i
            #         maxCount = bigColorNum[i]

            # for i in range(len(bigColorNum)):
            #     if (self.rgbEx[i] == "배경"):
            #             continue
            #     if (maxCount == bigColorNum[i]):
            #         count += 1

            # if (count >= 2):
            #     isCon2 = False
            # else:
            #     isCon2 = True

            # con2Color = self.largeColorEx[index]
            # print("con2: ", isCon2, " / con2Color : ",con2Color)

            # #con1 con2 비교
            # if (isCon1 and isCon2):
            #     self.finalColor = con2Color
            # elif (isCon2):
            #     self.finalColor = con2Color
            # elif (isCon1):
            #     self.finalColor = self.smallColorEx[index]
            # else:
            #     self.finalColor = "etc" #아무것도 해당되지 않음

            # 정확도 산출
            # count == imgAmount일 때 계산
            global correctCount
            global imgEnd
            imgEnd = imgEnd + 1
            if self.finalColor == imgColor:
                correctCount = correctCount + 1

            print("색상 결정 완료")
            print("imageId: ", imageId)
            print("con1: ", con1Color)
            print("con2: ", con2Color)
            print("finalColor: ", self.finalColor)

            return self.finalColor


        def excelChanged(self):
            # print("excelChange에서 줌심점 : ",self.standard)
            #RGB, HSV, 대분류 색상, 소분류 색상, 픽셀 개수, 이미지 차지 비율, k 값, 아이디, 총 픽셀 개수
            for i in range(self.k):
                if (np.array_equal(self.standard[i], [0, 0, 0])):
                    self.rgbEx[i] = "배경"
                    self.hsvEx[i] = "배경"
                    self.largeColorEx[i] = "배경"
                    self.smallColorEx[i] = "배경"
                    self.pixNumEx[i] = 0
                    self.pixRatioEx[i] = 0
                    continue
                #RGB, HSV 및 이미지 아이디
                self.rgbEx[i] = self.standard[i]
                self.hsvEx[i] = Algorithm.bgrToHsvOne(self.standard[i])

                #i번째 중심점의 index 값을 받아서 대분류된 색상 리턴해야함
                index = np.where(self.data == self.standard[i])[0][0]
                self.largeColorEx[i] = Algorithm.colorFigurationForExcel(index)
                self.smallColorEx[i] = Algorithm.colorFigurationForExcelSmall(index)
                #픽셀 개수
                self.pixNumEx[i] = self.eachCount[i]
                self.pixRatioEx[i] = self.pixNumEx[i] / (col*row) * 100
            
            for i in range(len(self.colorCount)):
                for j in range(self.k):
                    if (np.array_equal(self.standard[j], [0, 0, 0])):
                        continue
                    if (self.colorList[i] == self.largeColorEx[j]):
                        self.refColorCom[i] = self.refColorCom[i] + self.pixRatioEx[j]

            compare = 0
            for i in range(len(self.colorCount)):
                if (compare < self.refColorCom[i]):
                    compare = self.refColorCom[i]

            #self.refColorCom에서 compare 값의 index를 갖는 self.colorList의 값을 self.refColor에 저장
            refColor = self.colorList[self.refColorCom.index(compare)]
            for i in range(self.k):
                self.refColor[i] = refColor
            if ():
                excelData = {
                'Image Id' : self.imageId,
                'Number of K' : self.kNum,
                'RGB' : "배경",
                'HSV' : "배경",
                'Big Color' : "배경",
                'Small Color' : "배경",
                'Pixel Count' : self.pixNumEx,
                'Sum of Every Pixel' : self.pixNumSum,
                'Pixel Ratio' : self.pixRatioEx,
                'Most Color' : "배경",
                'Final Color' : self.colorDet()
                }
            else:
                excelData = {
                    'Image Id' : self.imageId,
                    'Number of K' : self.kNum,
                    'RGB' : self.rgbEx,
                    'HSV' : self.hsvEx,
                    'Big Color' : self.largeColorEx,
                    'Small Color' : self.smallColorEx,
                    'Pixel Count' : self.pixNumEx,
                    'Sum of Every Pixel' : self.pixNumSum,
                    'Pixel Ratio' : self.pixRatioEx,
                    'Most Color' : self.refColor,
                    'Final Color' : self.colorDet()
                }
            excelData = pd.DataFrame(excelData)
            # excelData.to_excel(excel_writer='sample.xlsx')

            if (not os.path.exists(fileName)):
                with pd.ExcelWriter(fileName, mode='w', engine='openpyxl') as writer:
                    excelData.to_excel(writer, sheet_name=sheetName)
            else:
                with pd.ExcelWriter(fileName, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                    excelData.to_excel(writer, sheet_name=sheetName, startcol=0, startrow=writer.sheets[sheetName].max_row)


        def accuracy(self):
            global colorAccuracy
            colorAccuracy = (correctCount / imgEnd) * 100


        def valueToText(self):
            # self.greenImage = self.resultCluster.copy()
            # self.redImage = self.resultCluster.copy()
            # self.blueImage = self.resultCluster.copy()
            # self.blueImage[:,:,1] = 0
            # self.blueImage[:,:,0] = 0
            # self.greenImage[:,:,0] = 0
            # self.greenImage[:,:,2] = 0
            # self.redImage[:,:,1] = 0
            # self.redImage[:,:,2] = 0
            hValueText = open("h.txt", 'w+')
            hToString = ''.join(str(self.hValue))
            hValueText.write(hToString)
            hValueText.close()
            sValueText = open("s.txt", 'w+')
            sToString = ''.join(str(self.sValue))
            sValueText.write(sToString)
            sValueText.close()
            vValueText = open("v.txt", 'w+')
            vToString = ''.join(str(self.vValue))
            vValueText.write(vToString)
            vValueText.close()
            # blueValueText = open("blue.txt", 'w+')
            # blueToString = ''.join(str(self.blueImage))
            # blueValueText.write(blueToString)
            # greenValueText = open("green.txt", 'w+')
            # greenToString = ''.join(str(self.greenImage))
            # greenValueText.write(greenToString)
            # redValueText = open("red.txt", 'w+')
            # redToString = ''.join(str(self.redImage))
            # redValueText.write(redToString)    
            # dataValueText = open("data.txt", 'w+')
            # dataToString = ''.join(str(self.data))
            # dataValueText.write(dataToString)


        def show(self):
            self.resultCluster = self.data.reshape((row, col, 3))
            resultImage = Image.fromarray(self.resultCluster.astype(np.uint8))
            # hImage = Image.fromarray(self.hValue.astype(np.uint8))
            # sImage = Image.fromarray(self.sValue.astype(np.uint8))
            # vImage = Image.fromarray(self.vValue.astype(np.uint8))
            # hsvImage = Image.fromarray((np.dstack((self.hValue,self.sValue,self.vValue)) * 255).astype(np.uint8))
            # hsvImage.show()
            # hImage.show()
            # sImage.show()
            # vImage.show()
            resultImage.show()


    for i in range(16, imgAmount+1):
        imgNum = str(i)
        print("현재 이미지 번호: ", imgNum, "번")

        #해당 이미지를 배열로 변환
        imgOrigin = imgAddress + imgColor + imgSlash + searchKey + imgNum + imgPng

        img = Image.open(imgOrigin) #오류 발생 : undefined Image, 해결 방법은 그냥 건너뛰기. 

        pix = np.array(img)
        imageId = imgColor + "_" + imgNum
        sheetName = imgColor + " Image"

        #3차원 배열 pix를 2차원 배열로 변환 뒤 floor 열 삭제
        # try:
        row,col,floor = pix.shape
        # if floor == 4:
        #     continue

        twoDim_array = np.column_stack(((np.repeat(np.arange(row), col)), pix.reshape(row*col, -1)))
        twoDim_array = np.delete(twoDim_array, 0, axis=1)
        # except:
        #     continue

        #알고리즘 실행
        Algorithm = K_Means(k, data=twoDim_array, row=row, col=col)
        Algorithm.clustering(5)
        Algorithm.quantization()
        Algorithm.bgrToHsv()
        Algorithm.colorFiguration()
        # Algorithm.valueToText()
        try:
            Algorithm.excelChanged()
        except:
            continue
        # Algorithm.show()

    Algorithm.accuracy()

    excelData = {
                'Image Id' : "x",
                'Number of K' : "x",
                'RGB' : "x",
                'HSV' : "x",
                'Big Color' : "x",
                'Small Color' : "x",
                'Pixel Count' : "x",
                'Sum of Every Pixel' : "x",
                'Pixel Ratio' : "x",
                'Most Color' : "x",
                'Final Color' : "x",
                'Accuracy' : colorAccuracy
            }

    imgAllCount[z] = imgEnd
    correctImgCount[z] = correctCount
    colorAccuracyList[z] = colorAccuracy

print(imgAllCount)
print(correctImgCount)
print(colorAccuracyList)