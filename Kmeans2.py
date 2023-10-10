from PIL import Image
import numpy as np
import random
import sys
import math
np.set_printoptions(threshold=sys.maxsize)


#해당 이미지를 배열로 변환
img = Image.open("/Users/chaewookim/Desktop/ColorClassification/blue/1.jpg")
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
        self.color = ["Color" for _ in range(self.N)] #세분화 되었을 때의 색
        self.colorCount = [0 for _ in range(12)] #각 색이 몇 개인지 확인하는 용도. 빨/주/노/연/녹/청/파/남/보/검/회/흰 순서
        self.colorList = ["Red", "Orange", "Yellow", "LightGreen", "Green", "Cyan", "Blue", "Indigo", "Magenta", "Black", "Grey", "White"]
        self.specificColorCount = [0 for _ in range(43)]
        self.specificColorList = ["Red(Y90R)", "Red(Y80R)", "Orange(Y70R)", "Orange(Y60R)", "Orange(Y50R)", "Orange(Y40R)", "Orange(Y30R)", "Orange(Y20R)", "Yellow(Y10R)", "Yellow(Y)", "Yellow(G90Y)", "Yellow(G80Y)", "LightGreen(G70Y)", "LightGreen(G60Y)", "LightGreen(G50Y)", "LightGreen(G40Y)", "LightGreen(G30Y)", "Green(G20Y)", "Green(G10Y)", "Green(G)", "Green(B90G)", "Green(B80G)", "Cyan(B70G)", "Cyan(B60G)", "Cyan(B50G)", "Cyan(B40G)", "Cyan(B30G)", "Cyan(B20G)", "Blue(B10G)", "Blue(B)", "Blue(R90B)", "Blue(R80B)", "Indigo(R70B)", "Indigo(R60B)", "Magenta(R50B)", "Magenta(R40B)", "Magenta(R30B)", "Magenta(R20B)", "Red(R10B)", "Red(R)", "Black", "Grey", "White"]


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
            self.hValue[input_row][input_col] = h
            self.sValue[input_row][input_col] = round(s*100)
            self.vValue[input_row][input_col] = round(v*100)

    
    #세분화된 색상을 넣어주는 함수
    def colorInput(self, specificColor, colorNumber, address):
        self.color[address] = specificColor
        self.colorCount[colorNumber] = self.colorCount[colorNumber] + 1
        self.specificColorCount[self.specificColorList.index(specificColor)] = self.specificColorCount[self.specificColorList.index(specificColor)] + 1


    #각 색을 12가지로 대분류하는 함수
    def red(self, address):
        if (self.hValue[int(address/col)][int(address%col)] > 343):
            Algorithm.colorInput("Red(Y90R)", 0, address)
        elif ((self.hValue[int(address/col)][int(address%col)] > 0) and (self.hValue[int(address/col)][int(address%col)] < 12)):
            Algorithm.colorInput("Red(Y80R)", 0, address)
        elif (self.hValue[int(address/col)][int(address%col)] == 337):
            Algorithm.colorInput("Red(R10B)", 0, address)
        elif ((self.hValue[int(address/col)][int(address%col)] > 337) and (self.hValue[int(address/col)][int(address%col)] < 344)):
            Algorithm.colorInput("Red(R)", 0, address)
    def orange(self, address):
        if ((self.hValue[int(address/col)][int(address%col)] > 11) and (self.hValue[int(address/col)][int(address%col)] < 20)):
            Algorithm.colorInput("Orange(Y70R)", 1, address)
        elif ((self.hValue[int(address/col)][int(address%col)] > 19) and (self.hValue[int(address/col)][int(address%col)] < 25)):
            Algorithm.colorInput("Orange(Y60R)", 1, address)
        elif ((self.hValue[int(address/col)][int(address%col)] > 24) and (self.hValue[int(address/col)][int(address%col)] < 30)):
            Algorithm.colorInput("Orange(Y50R)", 1, address)
        elif ((self.hValue[int(address/col)][int(address%col)] > 29) and (self.hValue[int(address/col)][int(address%col)] < 33)):
            Algorithm.colorInput("Orange(Y40R)", 1, address)
        elif ((self.hValue[int(address/col)][int(address%col)] > 32) and (self.hValue[int(address/col)][int(address%col)] < 37)):
            Algorithm.colorInput("Orange(Y30R)", 1, address)
        elif ((self.hValue[int(address/col)][int(address%col)] > 36) and (self.hValue[int(address/col)][int(address%col)] < 41)):
            Algorithm.colorInput("Orange(Y20R)", 1, address)
    def yellow(self, address):
        if ((self.hValue[int(address/col)][int(address%col)] > 40) and (self.hValue[int(address/col)][int(address%col)] < 44)):
            Algorithm.colorInput("Yellow(Y10R)", 2, address)
        elif ((self.hValue[int(address/col)][int(address%col)] > 43) and (self.hValue[int(address/col)][int(address%col)] < 51)):
            Algorithm.colorInput("Yellow(Y)", 2, address)
        elif ((self.hValue[int(address/col)][int(address%col)] > 50) and (self.hValue[int(address/col)][int(address%col)] < 54)):
            Algorithm.colorInput("Yellow(G90Y)", 2, address)
        elif ((self.hValue[int(address/col)][int(address%col)] > 53) and (self.hValue[int(address/col)][int(address%col)] < 56)):
            Algorithm.colorInput("Yellow(G80Y)", 2, address)
    def lightGreen(self, address):
        if ((self.hValue[int(address/col)][int(address%col)] > 42) and (self.hValue[int(address/col)][int(address%col)] < 57)):
            Algorithm.colorInput("LightGreen(G70Y)", 3, address)
        elif ((self.hValue[int(address/col)][int(address%col)] > 56) and (self.hValue[int(address/col)][int(address%col)] < 65)):
            Algorithm.colorInput("LightGreen(G60Y)", 3, address)
        elif ((self.hValue[int(address/col)][int(address%col)] > 64) and (self.hValue[int(address/col)][int(address%col)] < 68)):
            Algorithm.colorInput("LightGreen(G50Y)", 3, address)
        elif ((self.hValue[int(address/col)][int(address%col)] > 67) and (self.hValue[int(address/col)][int(address%col)] < 75)):
            Algorithm.colorInput("LightGreen(G40Y)", 3, address)
        elif ((self.hValue[int(address/col)][int(address%col)] > 74) and (self.hValue[int(address/col)][int(address%col)] < 89)):
            Algorithm.colorInput("LightGreen(G30Y)", 3, address)
    def green(self, address):
        if ((self.hValue[int(address/col)][int(address%col)] > 88) and (self.hValue[int(address/col)][int(address%col)] < 113)):
            Algorithm.colorInput("Green(G20Y)", 4, address)
        elif ((self.hValue[int(address/col)][int(address%col)] > 112) and (self.hValue[int(address/col)][int(address%col)] < 150)):
            Algorithm.colorInput("Green(G10Y)", 4, address)
        elif ((self.hValue[int(address/col)][int(address%col)] > 149) and (self.hValue[int(address/col)][int(address%col)] < 161)):
            Algorithm.colorInput("Green(G)", 4, address)
        elif ((self.hValue[int(address/col)][int(address%col)] > 160) and (self.hValue[int(address/col)][int(address%col)] < 166)):
            Algorithm.colorInput("Green(B90G)", 4, address)
        elif ((self.hValue[int(address/col)][int(address%col)] > 165) and (self.hValue[int(address/col)][int(address%col)] < 171)):
            Algorithm.colorInput("Green(B80G)", 4, address)
    def cyan(self, address):
        if ((self.hValue[int(address/col)][int(address%col)] > 170) and (self.hValue[int(address/col)][int(address%col)] < 173)):
            Algorithm.colorInput("Cyan(B70G)", 5, address)
        elif ((self.hValue[int(address/col)][int(address%col)] > 172) and (self.hValue[int(address/col)][int(address%col)] < 176)):
            Algorithm.colorInput("Cyan(B60G)", 5, address)
        elif ((self.hValue[int(address/col)][int(address%col)] > 175) and (self.hValue[int(address/col)][int(address%col)] < 178)):
            Algorithm.colorInput("Cyan(B50G)", 5, address)
        elif ((self.hValue[int(address/col)][int(address%col)] > 177) and (self.hValue[int(address/col)][int(address%col)] < 181)):
            Algorithm.colorInput("Cyan(B40G)", 5, address)
        elif ((self.hValue[int(address/col)][int(address%col)] > 180) and (self.hValue[int(address/col)][int(address%col)] < 184)):
            Algorithm.colorInput("Cyan(B30G)", 5, address)
        elif ((self.hValue[int(address/col)][int(address%col)] > 183) and (self.hValue[int(address/col)][int(address%col)] < 187)):
            Algorithm.colorInput("Cyan(B20G)", 5, address)
    def blue(self, address):
        if ((self.hValue[int(address/col)][int(address%col)] > 186) and (self.hValue[int(address/col)][int(address%col)] < 190)):
            Algorithm.colorInput("Blue(B10G)", 6, address)
        elif ((self.hValue[int(address/col)][int(address%col)] > 189) and (self.hValue[int(address/col)][int(address%col)] < 198)):
            Algorithm.colorInput("Blue(B)", 6, address)
        elif ((self.hValue[int(address/col)][int(address%col)] > 197) and (self.hValue[int(address/col)][int(address%col)] < 200)):
            Algorithm.colorInput("Blue(R90B)", 6, address)
        elif ((self.hValue[int(address/col)][int(address%col)] > 199) and (self.hValue[int(address/col)][int(address%col)] < 209)):
            Algorithm.colorInput("Blue(R80B)", 6, address)
    def indigo(self, address):
        if ((self.hValue[int(address/col)][int(address%col)] > 208) and (self.hValue[int(address/col)][int(address%col)] < 235)):
            Algorithm.colorInput("Indigo(R70B)", 7, address)
        elif ((self.hValue[int(address/col)][int(address%col)] > 234) and (self.hValue[int(address/col)][int(address%col)] < 260)):
            Algorithm.colorInput("Indigo(R60B)", 7, address)
    def magenta(self, address):
        if ((self.hValue[int(address/col)][int(address%col)] > 259) and (self.hValue[int(address/col)][int(address%col)] < 286)):
            Algorithm.colorInput("Magenta(R50B)", 8, address)
        elif ((self.hValue[int(address/col)][int(address%col)] > 285) and (self.hValue[int(address/col)][int(address%col)] < 314)):
            Algorithm.colorInput("Magenta(R40B)", 8, address)
        elif ((self.hValue[int(address/col)][int(address%col)] > 313) and (self.hValue[int(address/col)][int(address%col)] < 326)):
            Algorithm.colorInput("Magenta(R30B)", 8, address)
        elif ((self.hValue[int(address/col)][int(address%col)] > 325) and (self.hValue[int(address/col)][int(address%col)] < 337)):
            Algorithm.colorInput("Magenta(R20B)", 8, address)


    def colorFiguration(self):
        #여기서 색에 대한 11가지 분류를 잡고 각 함수를 호출해서 세분화
        for i in range(self.N):
            if (((self.sValue[int(i/col)][int(i%col)] <= 10) and (self.vValue[int(i/col)][int(i%col)] <= 30)) or ((self.sValue[int(i/col)][int(i%col)] <= 20) and (self.vValue[int(i/col)][int(i%col)] <= 30)) or ((self.sValue[int(i/col)][int(i%col)] <= 30) and (self.vValue[int(i/col)][int(i%col)] <= 40))):
                #Black
                Algorithm.colorInput("Black", 9, i)
            if ((self.sValue[int(i/col)][int(i%col)] <= 10) and (self.vValue[int(i/col)][int(i%col)] <= 70) and (self.vValue[int(i/col)][int(i%col)] > 30)):
                #Grey
                Algorithm.colorInput("Grey", 10, i)
            if ((self.hValue[int(i/col)][int(i%col)] == 0) and (self.sValue[int(i/col)][int(i%col)] == 0) and (self.vValue[int(i/col)][int(i%col)] == 100)):
                #White
                Algorithm.colorInput("White", 11, i)

            if ((self.hValue[int(i/col)][int(i%col)] < 12) or (self.hValue[int(i/col)][int(i%col)] > 336)):
                #Red
                if (self.color[i] != "Color"):
                    break
                Algorithm.red(i)
            elif (self.hValue[int(i/col)][int(i%col)] < 41):
                #Orange
                if (self.color[i] != "Color"):
                    break
                Algorithm.orange(i)
            elif (self.hValue[int(i/col)][int(i%col)] < 56):
                #Yellow
                if (self.color[i] != "Color"):
                    break
                Algorithm.yellow(i)
            elif (self.hValue[int(i/col)][int(i%col)] < 89):
                #LightGreen
                if (self.color[i] != "Color"):
                    break
                Algorithm.lightGreen(i)
            elif (self.hValue[int(i/col)][int(i%col)] < 171):
                #Green
                if (self.color[i] != "Color"):
                    break
                Algorithm.green(i)
            elif (self.hValue[int(i/col)][int(i%col)] < 187):
                #Cyan
                if (self.color[i] != "Color"):
                    break
                Algorithm.cyan(i)
            elif (self.hValue[int(i/col)][int(i%col)] < 209):
                #Blue
                if (self.color[i] != "Color"):
                    break
                Algorithm.blue(i)
            elif (self.hValue[int(i/col)][int(i%col)] < 260):
                #Indigo
                if (self.color[i] != "Color"):
                    break
                Algorithm.indigo(i)
            else:
                #Magenta
                if (self.color[i] != "Color"):
                    break
                Algorithm.magenta(i)
        
        colorText = open("colorText.txt", 'w+')
        colorToString = ''.join(str(self.color))
        colorText.write(colorToString)
        colorText.close()

        for i in range(len(self.colorCount)):
            print(self.colorList[i], " : ", self.colorCount[i])
        print("\n")
        for i in range(len(self.specificColorList)):
            print(self.specificColorList[i], " : ", self.specificColorCount[i])
        print(col*row)


    def valueToText(self):
    #     self.greenImage = self.resultCluster.copy()
    #     self.redImage = self.resultCluster.copy()
    #     self.blueImage = self.resultCluster.copy()
    #     self.blueImage[:,:,1] = 0
    #     self.blueImage[:,:,0] = 0
    #     self.greenImage[:,:,0] = 0
    #     self.greenImage[:,:,2] = 0
    #     self.redImage[:,:,1] = 0
    #     self.redImage[:,:,2] = 0
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
    #     blueValueText = open("blue.txt", 'w+')
    #     blueToString = ''.join(str(self.blueImage))
    #     blueValueText.write(blueToString)
    #     greenValueText = open("green.txt", 'w+')
    #     greenToString = ''.join(str(self.greenImage))
    #     greenValueText.write(greenToString)
    #     redValueText = open("red.txt", 'w+')
    #     redToString = ''.join(str(self.redImage))
    #     redValueText.write(redToString)    


    def show(self):
        self.resultCluster = self.data.reshape((pix.shape))
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
        
        
Algorithm = K_Means(k=3, data=twoDim_array, row=row, col=col)
Algorithm.clustering(4)
Algorithm.quantization()
Algorithm.bgrToHsv()
Algorithm.colorFiguration()
Algorithm.valueToText()
Algorithm.show()