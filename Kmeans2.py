from PIL import Image
import numpy as np
import random
import sys
import math
np.set_printoptions(threshold=sys.maxsize)


#해당 이미지를 배열로 변환
img = Image.open("/Users/chaewookim/Desktop/ColorClassification/Blue/4.jpg")
pix = np.array(img)
k=5


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
        self.bgr = [[0,0,0] for _ in range(self.k)]
        self.hsv = [[0,0,0] for _ in range(self.k)]

        self.eachCount = [0 for _ in range(self.k)]

        self.backgroundColor = [0 for _ in range(self.k-1)]

        #Color
        self.color = ["Color" for _ in range(self.N)] #세분화 되었을 때의 색
        self.colorCount = [0 for _ in range(12)] #각 색이 몇 개인지 확인하는 용도. 빨/주/노/연/녹/청/파/남/보/검/회/흰 순서
        self.colorList = ["Red", "Orange", "Yellow", "LightGreen", "Green", "Cyan", "Blue", "Indigo", "Magenta", "Black", "Grey", "White"]
        self.redCount = [0 for _ in range(4)]
        self.orangeCount = [0 for _ in range(6)]
        self.yellowCount = [0 for _ in range(4)]
        self.lightGreenCount = [0 for _ in range(5)]
        self.greenCount = [0 for _ in range(5)]
        self.cyanCount = [0 for _ in range(6)]
        self.blueCount = [0 for _ in range(4)]
        self.indigoCount = [0 for _ in range(2)]
        self.magentaCount = [0 for _ in range(4)]
        self.blackCount = [0]
        self.greyCount = [0]
        self.whiteCount = [0]
        self.allColorCount = [self.redCount, self.orangeCount, self.yellowCount, self.lightGreenCount, self.greenCount, self.cyanCount, self.blueCount, self.indigoCount, self.magentaCount, self.blackCount, self.greyCount, self.whiteCount]
        #ColorType
        self.redColor = ["Red(R10B)", "Red(R)", "Red(Y90R)", "Red(Y80R)"]
        self.orangeColor = ["Orange(Y70R)", "Orange(Y60R)", "Orange(Y50R)", "Orange(Y40R)", "Orange(Y30R)", "Orange(Y20R)"]
        self.yellowColor = ["Yellow(Y10R)", "Yellow(Y)", "Yellow(G90Y)", "Yellow(G80Y)"]
        self.lightGreenColor = ["LightGreen(G70Y)", "LightGreen(G60Y)", "LightGreen(G50Y)", "LightGreen(G40Y)", "LightGreen(G30Y)"]
        self.greenColor = ["Green(G20Y)", "Green(G10Y)", "Green(G)", "Green(B90G)", "Green(B80G)"]
        self.cyanColor = ["Cyan(B70G)", "Cyan(B60G)", "Cyan(B50G)", "Cyan(B40G)", "Cyan(B30G)", "Cyan(B20G)"]
        self.blueColor = ["Blue(B10G)", "Blue(B)", "Blue(R90B)", "Blue(R80B)"]
        self.indigoColor = ["Indigo(R70B)", "Indigo(R60B)"]
        self.magentaColor = ["Magenta(R50B)", "Magenta(R40B)", "Magenta(R30B)", "Magenta(R20B)"]
        self.blackColor = ["Black"]
        self.greyColor = ["Grey"]
        self.whiteColor = ["White"]
        self.allColor = [self.redColor, self.orangeColor, self.yellowColor, self.lightGreenColor, self.greenColor, self.cyanColor, self.blueColor, self.indigoColor, self.magentaColor, self.blackColor, self.greyColor, self.whiteColor]
        #StartColor
        self.redStartColor = [337, 338, 346, 1] 
        self.orangeStartColor = [12, 20, 25, 30, 33, 37]
        self.yellowStartColor = [41, 44, 51, 54]
        self.lightGreenStartColor = [56, 61, 65, 68, 75]
        self.greenStartColor = [89, 113, 150, 161, 166]
        self.cyanStartColor = [171, 173, 176, 178, 181, 184]
        self.blueStartColor = [187, 190, 198, 200]
        self.indigoStartColor = [209, 235]
        self.magentaStartColor = [260, 286, 314, 326]
        self.allStartColor = [self.redStartColor, self.orangeStartColor, self.yellowStartColor, self.lightGreenStartColor, self.greenStartColor, self.cyanStartColor, self.blueStartColor, self.indigoStartColor, self.magentaStartColor]
        #EndColor
        self.redEndColor = [337, 345, 0, 11]
        self.orangeEndColor = [19, 24, 29, 32, 36, 40]
        self.yellowEndColor = [43, 50, 53, 55]
        self.lightGreenEndColor = [60, 64, 67, 74, 88]
        self.greenEndColor = [112, 149, 160, 165, 170]
        self.cyanEndColor = [172, 175, 177, 180, 183, 186]
        self.blueEndColor = [189, 197, 199, 208]
        self.indigoEndColor = [234, 259]
        self.magentaEndColor = [285, 313, 325, 336]
        self.allEndColor = [self.redEndColor, self.orangeEndColor, self.yellowEndColor, self.lightGreenEndColor, self.greenEndColor, self.cyanEndColor, self.blueEndColor, self.indigoEndColor, self.magentaEndColor]


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
        for i in range(self.N):
            for j in range(self.k):
                if (self.Cluster[i] == j):
                    self.data[i] = self.standard[j]
                    self.eachCount[j] = self.eachCount[j]+1
        self.resultCluster = self.data.reshape((pix.shape))


    def bgrToHsv(self):
        #bgr을 hsv로 변환
        for i in range(self.N):
            maximum, minimum = 0, 0
            r = float(self.data[i][0])/255
            g = float(self.data[i][1])/255
            b = float(self.data[i][2])/255
            maximum = max(r,g,b)
            minimum = min(r,g,b)

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
        maximum, minimum = 0, 0
        r = float(input[0])/255
        g = float(input[1])/255
        b = float(input[2])/255
        maximum = max(r,g,b)
        minimum = min(r,g,b)

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
                
        return [round(h), round(s*100), round(v*100)]


    def colorInput(self, index, colorNumber, nth):
        #1. 색 소분류 결과 값 저장
        #2. 색 대분류 카운트 증가
        #3. 색 소분류 카운트 증가
        #검회흰색을 넣고 이중으로 카운트 하지 않기 위한 조건문 추가
        if (self.color[index] == "Color"):
            self.color[index] = self.allColor[colorNumber][nth]
            self.colorCount[colorNumber] = self.colorCount[colorNumber] + 1
            self.allColorCount[colorNumber][nth] = self.allColorCount[colorNumber][nth] + 1


    #색 소분류
    def colorSmallClassify(self, index, colorNumber):
        for i in range(len(self.allColor[colorNumber])):
            if ((self.hValue[int(index/col)][int(index%col)] > (self.allStartColor[colorNumber][i] - 1)) and (self.hValue[int(index/col)][int(index%col)] <= self.allEndColor[colorNumber][i])):
                Algorithm.colorInput(index, colorNumber, i)
                break
            #hValue가 346~0의 범위에 있을 때
            if ((colorNumber == 0) and (self.allStartColor[colorNumber][i] == 1)):
                if ((self.hValue[int(index/col)][int(index%col)] >= (self.allStartColor[colorNumber][i] - 1)) or (self.hValue[int(index/col)][int(index%col)] <= self.allEndColor[colorNumber][i])):
                    Algorithm.colorInput(index, colorNumber, i)

    #색 대분류
    def colorLargeClassify(self, index):
        if (self.hValue[int(index/col)][int(index%col)] < self.cyanStartColor[0]):
            if (self.hValue[int(index/col)][int(index%col)] < self.yellowStartColor[0]):
                if (self.hValue[int(index/col)][int(index%col)] < self.orangeStartColor[0]):
                    #Red
                    Algorithm.colorSmallClassify(index, 0)
                else:
                    #Orange
                    Algorithm.colorSmallClassify(index, 1)
            else:
                if (self.hValue[int(index/col)][int(index%col)] < self.greenStartColor[0]):
                    if (self.hValue[int(index/col)][int(index%col)] < self.lightGreenStartColor[0]):
                        #Yellow
                        Algorithm.colorSmallClassify(index, 2)
                    else:
                        #LightGreen
                        Algorithm.colorSmallClassify(index, 3)
                else:
                    #green
                    Algorithm.colorSmallClassify(index, 4)
        else:
            if (self.hValue[int(index/col)][int(index%col)] < self.indigoStartColor[0]):
                if (self.hValue[int(index/col)][int(index%col)] < self.blueStartColor[0]):
                    #Cyan
                    Algorithm.colorSmallClassify(index, 5)
                else:
                    #Blue
                    Algorithm.colorSmallClassify(index, 6)
            else:
                if (self.hValue[int(index/col)][int(index%col)] < self.magentaStartColor[0]):
                    #indigo
                    Algorithm.colorSmallClassify(index, 7)
                else:
                    if (self.hValue[int(index/col)][int(index%col)] <= self.magentaEndColor[len(self.magentaEndColor)-1]):
                        #Magenta
                        Algorithm.colorSmallClassify(index, 8)
                    else:
                        #Red
                        Algorithm.colorSmallClassify(index, 0)


    def colorFiguration(self):
        #대분류, 소분류 함수들 호출
        for i in range(self.N):
            #Black
            if ((self.sValue[int(i/col)][int(i%col)] <= 10) and (self.vValue[int(i/col)][int(i%col)] <= 30)):
                Algorithm.colorInput(i, 9, 0)
            if ((self.sValue[int(i/col)][int(i%col)] <= 20) and (self.sValue[int(i/col)][int(i%col)] > 10) and (self.vValue[int(i/col)][int(i%col)] <= 30)):
                Algorithm.colorInput(i, 9, 0)
            if ((self.sValue[int(i/col)][int(i%col)] <= 30) and (self.sValue[int(i/col)][int(i%col)] > 20) and (self.vValue[int(i/col)][int(i%col)] <= 40)):
                Algorithm.colorInput(i, 9, 0)

            #Grey            
            if ((self.sValue[int(i/col)][int(i%col)] <= 10) and (self.vValue[int(i/col)][int(i%col)] <= 70) and (self.vValue[int(i/col)][int(i%col)] > 30)):
                Algorithm.colorInput(i, 10, 0)

            #White
            if ((self.sValue[int(i/col)][int(i%col)] == 0) and (self.vValue[int(i/col)][int(i%col)] == 100)):
                Algorithm.colorInput(i, 11, 0)

            #나머지 색들은 s,v를 고려하지 않기 때문에 검흰회색보다 나중에 실행
            Algorithm.colorLargeClassify(i)

        
        colorText = open("colorText.txt", 'w+')
        colorToString = ''.join(str(self.color))
        colorText.write(colorToString)
        colorText.close()

        sum = 0
        print("  R  G  B")
        for i in range(self.k):
            print(self.standard[i], " : ", self.eachCount[i])
            sum = sum + self.eachCount[i]

        print("----------------")
        print("  H  S  V")
        for i in range(self.k):
            print(Algorithm.bgrToHsvOne(self.standard[i]))

        print("------------------------------------")        
        for i in range(len(self.colorCount)):
            print(self.colorList[i], " : ", self.colorCount[i])
        print("------------------------------------")
        for i in range(len(self.colorList)):
            for j in range(len(self.allColorCount[i])):
                print(self.allColor[i][j], " : ", self.allColorCount[i][j])
        print("------------------------------------")
        print("합 : ",sum)
        print("총 개수 : ", col*row)


    def backgroundDelete(self):
        background = (100,100,100,0)

        self.resultCluster = self.data.reshape((pix.shape))
        resultImage = Image.fromarray(self.resultCluster.astype(np.uint8))

        self.newData = resultImage.convert("RGBA")

        for i in range(row):
            for j in range(col):
                if (self.color[i*row + col] == "White"):
                    self.newData[i][j] = background


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


    def show(self):
        # self.resultCluster = self.data.reshape((pix.shape))
        # resultImage = Image.fromarray(self.resultCluster.astype(np.uint8))
        # hImage = Image.fromarray(self.hValue.astype(np.uint8))
        # sImage = Image.fromarray(self.sValue.astype(np.uint8))
        # vImage = Image.fromarray(self.vValue.astype(np.uint8))
        # hsvImage = Image.fromarray((np.dstack((self.hValue,self.sValue,self.vValue)) * 255).astype(np.uint8))
        # hsvImage.show()
        # hImage.show()
        # sImage.show()
        # vImage.show()
        self.newData.show()
        
        
Algorithm = K_Means(k, data=twoDim_array, row=row, col=col)
Algorithm.clustering(4)
Algorithm.quantization()
Algorithm.bgrToHsv()
Algorithm.colorFiguration()
Algorithm.valueToText()

Algorithm.backgroundDelete()

Algorithm.show()