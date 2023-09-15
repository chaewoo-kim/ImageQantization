from PIL import Image
import numpy as np
import pandas as pd
import openpyxl
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

#해당 이미지를 배열로 변환
img = Image.open("rgb.jpg")
pix = np.array(img)

#3차원 배열 pix를 2차원 배열로 변환 뒤 데이터프레임 df로 변환 
floor,row,col = pix.shape
twoDim_array = np.column_stack(((np.repeat(np.arange(floor), row)), pix.reshape(floor*row, -1)))
df = pd.DataFrame(twoDim_array, columns=['floor', 'R', 'G', 'B'])
df = df.drop('floor', axis=1) #3차원 배열 때 층수 나타낸 floor 열 제거

print(df)
'''
class K_Means:
    #k와 학습시킬 data 받아오기
    #__init__은 반드시 첫 매개변수를 self로 해야함
    def __init__(self,k,data): #초기화 함수, 객체 생성 시 반드시 처음 호출.
        self.k = k
        self.data = data
        self.N = len(data) #데이터의 길이, 즉 데이터의 개수
        self.standard = [] #초기 centroid 저장 리스트
        self.Cluster = np.array([0 for _ in range(self.N)]) #1행 N열의 shape 갖는 data들의 값 저장. 변수 없이 반복문 실행하고 싶을 때 _를 반복문의 변수 자리에 입력 
        #Cluster = [0, 0, --- , 0, 0]의 모양새. self.Cluster.shape == self.N

        #k개의 centroid 무작위로 선정
        while(len(self.standard) < k):
            #동일한 centroid 값이 나오지 않도록 random 이용
            tmp = np.random.randint(self.N) 
            if tmp not in self.standard: 
                self.standard.append(tmp) #standard 리스트에 tmp 추가
                self.Cluster[tmp] = len(self.standard) #이상함. tmp 자리에 self.standard가 들어가서 군집의 개수가 조절되어야 하는 것이 아닌지?

        for i in range(k):
            self.standard[i] = self.data[self.standard[i]] 

    def distance(self,centroid,data): #centroid와 데이터 간의 거리 구하는 함수
        #거리는 유클리드 거리를 사용한다고 가정
        return sum(abs(centroid-data)**2)**(0.5)
    
    def Clustering(self,clustering_count): #clstering_count는 클러스터링 진행 횟수
        for _ in range(clustering_count): #clustering_count의 횟수만큼 반복
            original_Cluster = np.array(self.Cluster) #비교를 위해 변경되기 클러스터링 진행 이전의 Cluster 저장 

            #모든 데이터의 각 centroid들로부터의 최소 거리+1을 Cluster 배열에 저장
            for i in range(self.N): #데이터 개수만큼 반복
                dist = [] 
                for standard in self.standard:
                    dist.append(self.distance(self.data[i],standard)) #각 centroid와 i번째 데이터 간의 거리를 dist 리스트에 저장 
                self.Cluster[i] = np.argmin(dist) + 1 #centroid들과의 거리 중 최소값+1을 Cluster[i]에 저장. np.argmin()은 최소값의 index 값 반환 

            #
            Clusters = [[] for _ in range(self.k+1)] #데이터들을 모아놓는 군집덜의 리스트. centroid의 개수+1 만큼의 공간 갖는 리스트
            for i in range(self.N):
                Clusters[self.Cluster[i]].append(self.data[i])
            for i in range(1,self.k+1):
                Clusters[i] = np.array(Clusters[i])
                self.standard[i-1] = Clusters[i].mean(axis=0)
            
            #Cluster의 변화가 없을 때 종료
            if np.array_equal(original_Cluster,self.Cluster): #두 배열을 비교해 같을 시에만 True 리턴
                break


Algorithm = K_Means(k=4, data=df)
Algorithm.Clustering()


'''