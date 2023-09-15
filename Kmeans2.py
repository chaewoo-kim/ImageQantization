from PIL import Image
import numpy as np
import pandas as pd
import random
import math
import colorsys


#해당 이미지를 배열로 변환
img = Image.open("yellow_pill.jpg")
#yellow_pill은 427행 640열
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
        self.centroid = [int((row-1)/2), int((col-1)/2)] #기준이 되어주는 사진의 중심
        self.centroidList = [0 for _ in range(self.k)] #중심점들의 self.data에서의 주소 저장
        self.distanceList = [0 for _ in range(self.k)] #데이터들의 각 중심점으로부터의 거리를 저장하는 리스트
        self.centroid_row = [0 for _ in range(self.k)] #중심점의 row 값을 저장하는 크기가 데이터 k만큼인 리스트
        self.centroid_col = [0 for _ in range(self.k)] #중심점의 col 값을 저장하는 크기가 데이터 k만큼인 리스트
        self.Cluster = [0 for _ in range(self.N)] #각 데이터들이 속해있는 군집의 중심점 값을 저장하는 리스트
        self.resultCluster = np.array([0 for _ in range(self.N)]) #색상 양자화를 진행 후 화면 출력에 사용할 배열
        self.hCluster = np.array([0 for _ in range(self.N)]) #hsv 중 h로 출력에 사용할 배열
        self.sCluster = np.array([0 for _ in range(self.N)]) #hsv 중 s로 출력에 사용할 배열
        self.vCluster = np.array([0 for _ in range(self.N)]) #hsv 중 v로 출력에 사용할 배열

        self.arrr = np.array([0 for _ in range(self.k)])


        '''
        <중심점 설정 방식>
        1. 무작위로 하나의 점 설정
        2. 1st 중심점과 가장 먼 점의 중간 점을 2nd 중심점으로 설정
        3. 1st, 2nd 두 점의 무게중심과 가장 먼 점의 중간 점을 3rd 중심점으로 설정
        4. 1st, 2nd, 3rd 세 점의 무게중심과 가장 먼 점의 중간 점을 4th 중심점으로 설정
        5. 반복

        <구체화>
        첫 중심점의 값을 self.standard에 저장하면서 좌표의 주소를 self.centroid_address에 저장. self.data[tmp]의 주소를 col로 나눠서 몫이 행, 나머지가 열.
        첫 중심점의 위치와 정반대의 꼭짓점이 두 번째 중심점. 두 번째 중심점까지만 수동으로 정하고 이후부터는 무게중심으로 중심점을 설정. k-2번 반복
        '''
        
        #0번째부터 시작하기 위해 row, col 모두 1씩 감소
        row -= 1
        col -= 1
        #첫 중심점 정하기
        #list, count, cluster 채우기
        if (self.k == 3):
            self.centroid_row[0] = self.centroid[0]/3*4
            self.centroid_col[0] = self.centroid[1]/3
            self.centroid_row[1] = self.centroid[0]/3*4
            self.centroid_col[1] = self.centroid[1]/3*5
            self.centroid_row[2] = self.centroid[0]/3
            self.centroid_col[2] = self.centroid[1]
            arr = np.array([0 for _ in range(self.k)])
            for i in range(self.k):
                self.centroidList[i] = int(self.centroid_row[i]*col + self.centroid_col[i])
                arr[i] = self.centroidList[i]
                self.standard[i] = self.data[arr[i]]
        if (self.k == 4):
            self.centroid_row[0] = self.centroid[0]/3*4
            self.centroid_col[0] = self.centroid[1]/3
            self.centroid_row[1] = self.centroid[0]/3*4
            self.centroid_col[1] = self.centroid[1]/3*5
            self.centroid_row[2] = self.centroid[0]/3
            self.centroid_col[2] = self.centroid[1]
            self.centroid_row[3] = self.centroid[0]
            self.centroid_row[3] = self.centroid[1]
            arr = np.array([0 for _ in range(self.k)])
            for i in range(self.k):
                self.centroidList[i] = int(self.centroid_row[i]*col + self.centroid_col[i])
                arr[i] = self.centroidList[i]
                self.standard[i] = self.data[arr[i]]
        if (self.k == 5):
            self.centroid_row[0] = self.centroid[0]/3*5
            self.centroid_col[0] = self.centroid[1]/3
            self.centroid_row[1] = self.centroid[0]/3*5
            self.centroid_col[1] = self.centroid[1]/3*5
            self.centroid_row[2] = self.centroid[0]/3
            self.centroid_col[2] = self.centroid[1]/3
            self.centroid_row[3] = self.centroid[0]/3
            self.centroid_col[3] = self.centroid[1]/3*5
            self.centroid_col[4] = self.centroid[0]
            self.centroid_row[4] = self.centroid[1]
            arr = np.array([0 for _ in range(self.k)])
            for i in range(self.k):
                self.centroidList[i] = int(self.centroid_row[i]*col + self.centroid_col[i])
                arr[i] = self.centroidList[i]
                self.standard[i] = self.data[arr[i]]
        if (self.k == 6):
            self.centroid_row[0] = self.centroid[0]
            self.centroid_row[0] = self.centroid[1]/3
            self.centroid_row[1] = self.centroid[0]
            self.centroid_col[1] = self.centroid[1]/3*5
            self.centroid_row[2] = self.centroid[0]/3*5
            self.centroid_col[2] = self.centroid[1]/3*2
            self.centroid_row[3] = self.centroid[0]/3*5
            self.centroid_col[3] = self.centroid[1]/3*5
            self.centroid_row[4] = self.centroid[0]/3
            self.centroid_col[4] = self.centroid[1]
            self.centroid_row[5] = self.centroid[0]
            self.centroid_col[5] = self.centroid[1]
            arr = np.array([0 for _ in range(self.k)])
            for i in range(self.k):
                self.centroidList[i] = int(self.centroid_row[i]*col + self.centroid_col[i])
                arr[i] = self.centroidList[i]
                self.standard[i] = self.data[arr[i]]
        if (self.k == 7):
            self.centroid_row[0] = self.centroid[0]
            self.centroid_row[0] = self.centroid[1]/3
            self.centroid_row[1] = self.centroid[0]
            self.centroid_col[1] = self.centroid[1]/3*5
            self.centroid_row[2] = self.centroid[0]/3*5
            self.centroid_col[2] = self.centroid[1]/3*2
            self.centroid_row[3] = self.centroid[0]/3*5
            self.centroid_col[3] = self.centroid[1]/3*5
            self.centroid_row[4] = self.centroid[0]/3
            self.centroid_col[4] = self.centroid[1]/3*2
            self.centroid_row[5] = self.centroid[0]/3
            self.centroid_col[5] = self.centroid[1]/3*4
            self.centroid_row[6] = self.centroid[0]
            self.centroid_col[6] = self.centroid[1]
            arr = np.array([0 for _ in range(self.k)])
            for i in range(self.k):
                self.centroidList[i] = int(self.centroid_row[i]*col + self.centroid_col[i])
                arr[i] = self.centroidList[i]
                self.standard[i] = self.data[arr[i]]
        if (self.k == 8):
            self.centroid_row[0] = self.centroid[0]/3
            self.centroid_col[0] = self.centroid[1]/3*2
            self.centroid_row[1] = self.centroid[0]/3
            self.centroid_col[1] = self.centroid[1]/3*4
            self.centroid_row[2] = self.centroid[0]/3*2
            self.centroid_col[2] = self.centroid[1]/3
            self.centroid_row[3] = self.centroid[0]/3*2
            self.centroid_col[3] = self.centroid[1]/3*5
            self.centroid_row[4] = self.centroid[0]/3*4
            self.centroid_col[4] = self.centroid[1]/3
            self.centroid_row[5] = self.centroid[0]/3*4
            self.centroid_col[5] = self.centroid[1]/3*5
            self.centroid_row[6] = self.centroid[0]/3*5
            self.centroid_col[6] = self.centroid[1]
            self.centroid_row[7] = self.centroid[0]
            self.centroid_col[7] = self.centroid[1]
            arr = np.array([0 for _ in range(self.k)])
            for i in range(self.k):
                self.centroidList[i] = int(self.centroid_row[i]*col + self.centroid_col[i])
                arr[i] = self.centroidList[i]
                self.standard[i] = self.data[arr[i]]
        if (self.k == 9):
            self.centroid_row[0] = int(self.centroid[0]/3)
            self.centroid_col[0] = int(self.centroid[1]/3*2)
            self.centroid_row[1] = int(self.centroid[0]/3)
            self.centroid_col[1] = int(self.centroid[1]/3*4)
            self.centroid_row[2] = int(self.centroid[0]/3*2)
            self.centroid_col[2] = int(self.centroid[1]/3)
            self.centroid_row[3] = int(self.centroid[0]/3*2)
            self.centroid_col[3] = int(self.centroid[1]/3*5)
            self.centroid_row[4] = int(self.centroid[0]/3*4)
            self.centroid_col[4] = int(self.centroid[1]/3)
            self.centroid_row[5] = int(self.centroid[0]/3*4)
            self.centroid_col[5] = int(self.centroid[1]/3*5)
            self.centroid_row[6] = int(self.centroid[0]/3*5)
            self.centroid_col[6] = int(self.centroid[1]/3*2)
            self.centroid_row[7] = int(self.centroid[0]/3*5)
            self.centroid_col[7] = int(self.centroid[1]/3*5)
            self.centroid_row[8] = int(self.centroid[0])
            self.centroid_col[8] = int(self.centroid[1])
            arr = np.array([0 for _ in range(self.k)])
            for i in range(self.k):
                self.centroidList[i] = int(self.centroid_row[i]*col + self.centroid_col[i])
                arr[i] = self.centroidList[i]
                self.standard[i] = self.data[arr[i]][:]


    #클러스터링 하는 함수
    def clustering(self,clusteringCount): 
        for _ in range(clusteringCount): 
            #군집화 과정
            #1. 데이터와 중심점 간의 거리 측정해 distanceList에 저장
            #1-1. 거리는 인덱스를 기준으로 행렬로 바꿔서 얼마나 가까이 있는지 책정
            #2. 가장 가까운 중심점의 index를 self.Cluster[i]에 저장
            #3. 모든 점에 대해 전부 반복
            self.distanceList = [0 for _ in range(self.k)] #데이터들의 각 중심점으로부터의 거리를 저장하는 리스트
            self.centroidCount = [0 for _ in range(self.k)] #각 군집에 데이터가 몇 개 속해있는지 카운트하는 리스트
            for i in range(self.N):
                for j in range(self.k):
                    #해당 점의 행렬과 중심점의 행렬 사이의 거리 측정
                    dataRow, dataCol = 0, 0
                    dataRow = int((int(i/col) - self.centroid_row[j])**2)
                    dataCol = int((int(i%col) - self.centroid_col[j])**2)
                    self.distanceList[j] = (dataRow + dataCol)
                    #self.distanceList의 값들을 비교해 가장 작은 값의 index가 그 데이터가 속하는 중심점
                    #self.Cluster의 index는 self.data의 index와 완전히 같음
                    #또한 self.Cluster의 값은 해당 index의 self.data가 어느 중심점에 속해있는지 저장되어 있음
                minmin = self.distanceList[0]
                for j in range(self.k):
                    if (minmin > self.distanceList[j]):
                        minmin = self.distanceList[j]
                self.Cluster[i] = self.distanceList.index(minmin)
                self.centroidCount[self.distanceList.index(minmin)] += 1


            #중심점 재설정
            #중심값은 중심점까지의 거리의 평균으로 재설정. 행렬을 기준으로 왼쪽과 하단을 -, 오른쪽과 상단을 +로 간주
            #1. 필요 변수 혹은 리스트 : 각 중심점 별로 행, 열의 합을 저장할 변수
            for i in range(self.k):
                self.centroidRowDisList = [0 for _ in range(self.k)] #각 요소에 중심점으로부터 행으로 떨어진 평균 길이 저장
                self.centroidColDisList = [0 for _ in range(self.k)] #각 요소에 중심점으로부터 열로 떨어진 평균 길이 저장
                rowAverage = [0 for _ in range(self.k)]
                colAverage = [0 for _ in range(self.k)]
                for j in range(self.N):
                    if (self.Cluster[j] == i):
                        self.centroidRowDisList[i] += (j/col) - self.centroid_row[i]
                        self.centroidColDisList[i] += (j%col) - self.centroid_col[i]
                rowAverage[i] = int(self.centroidRowDisList[i]/self.centroidCount[i])
                colAverage[i] = int(self.centroidColDisList[i]/self.centroidCount[i])
                if ((rowAverage[i] + self.centroidList[i]/col)*col + colAverage[i] + self.centroidList[i]%col < row*col + col):
                    self.centroidList[i] = int((rowAverage[i] + self.centroidList[i]/col)*col + colAverage[i] + self.centroidList[i]%col)
                    self.arrr[i] = self.centroidList[i]
                    self.standard[i] = self.data[self.arrr[i]][:]

            # print(self.standard)
            # print(self.centroidList)
            # print(self.centroidCount)
            # print(self.centroidList)
            # print(self.standard)
        

    def quantization(self):
        self.datad = self.data.copy()
        #색상 양자화
        #데이터의 값을 가장 가까운 중심점의 값으로 변경
        print(self.standard)
        print(self.data[0])
        for i in range(self.N):
            for j in range(self.k):
                if (self.Cluster[i] == j):
                    self.datad[i] = self.standard[j]
        print(self.standard)
        self.resultCluster = self.datad.reshape((pix.shape))
        
        
        
        

    def show(self):
        resultImage = Image.fromarray(self.resultCluster.astype(np.uint8))
        resultImage.show()

        """

        # HImage = Image.fromarray(self.H.astype(np.uint8))
        # HImage.show()
        # SImage = Image.fromarray(self.S.astype(np.uint8))
        # SImage.show()
        # VImage = Image.fromarray(self.V.astype(np.uint8))
        # VImage.show()

        """

        
        
Algorithm = K_Means(k=7, data=twoDim_array, row=row, col=col)
Algorithm.clustering(4)
Algorithm.quantization()
Algorithm.show()
