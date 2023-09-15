import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

current_path = os.path.dirname(os.path.abspath(__file__)) #파일 불러읽기
raw_data = pd.read_csv('Mall_Customers.csv',encoding='utf-8', engine = 'python')
Customers_data = raw_data[['Gender','Age','Annual Income (k$)','Spending Score (1-100)']].to_numpy()    # x 데이터 받기
for i in range(len(Customers_data)): #남성,여성을 숫자로 생각할 수 있게
    if Customers_data[i][0] =='Male':
        Customers_data[i][0] = 1
    else:
        Customers_data[i][0] = 0
Annual_Income = raw_data[['Annual Income (k$)']].to_numpy()
Spending_Score = raw_data[['Spending Score (1-100)']].to_numpy()

plt.scatter(Annual_Income, Spending_Score,s=50,c='violet',marker = 'o', label = 'Centroids')
plt.legend()
plt.title(f'results',size = 15)
plt.xlabel('Annual Income',size = 12)
plt.ylabel('Spending Score',size = 12)
plt.show() #데이터 그래프 표시

class K_Means: #K-Means 알고리즘을 클래스로 구현
    def __init__(self,k,data,p): #k,학습시킬데이터,p를 받아옴
        self.k = k
        self.data = data
        self.p = p
        self.N = len(data)
        self.standard = []
        self.Cluster = np.array([0 for _ in range(self.N)])#군집을 저장하는 공간
        while(len(self.standard) < k): #k값에 따라 처음값 설정
            tmp = np.random.randint(self.N) #중복값은 설정 안되게설정
            if tmp not in self.standard:
                self.standard.append(tmp)
                self.Cluster[tmp] = len(self.standard) #초기군집을 1,2,3으로 넣어줌

        for i in range(k): #계산하기 편하게 처음 세개 값을 index가 아닌 실제값으로 변환
            self.standard[i] = self.data[self.standard[i]]

    def distance(self,I,J): #거리 구하는 함수
        return sum(abs(I-J)**self.p)**(1/self.p)

    def Clustering(self): #Clustering 알고리즘
        for _ in range(100): ##iteration 100회로 임의 설정
            Old_Cluster = np.array(self.Cluster)
            for i in range(self.N):
                dist = []
                for standard in self.standard:
                    dist.append(self.distance(self.data[i],standard))
                self.Cluster[i] = np.argmin(dist) + 1

            Clusters = [[] for __ in range(self.k + 1)]
            for i in range(self.N):
                Clusters[self.Cluster[i]].append(self.data[i])
            for i in range(1,self.k+1):
                Clusters[i] = np.array(Clusters[i])
                self.standard[i-1] = Clusters[i].mean(axis=0)

            #cluster가 변화없어질때 종료조건을 넣을 수 있다.
            # if np.array_equal(Old_Cluster,self.Cluster):
            #     break



Algorism = K_Means(k = 4,data = Customers_data,p=1) #모델 구현
Algorism.Clustering()#Clustering 실행
Annual = [ [] for _ in range(Algorism.k+1)] #값을 볼 수 있는 형태로 변환
Spending = [ [] for _ in range(Algorism.k+1)]
for i in range(Algorism.N): #같은 군집끼리 값을 모은다
    Annual[Algorism.Cluster[i]].append(Annual_Income[i])
    Spending[Algorism.Cluster[i]].append(Spending_Score[i])

for i in range(1,Algorism.k+1): # k개의 군집 출력
    plt.scatter(Annual[i], Spending[i],s=50,marker = 'o', label = f'cluster{i}')
    plt.scatter(Algorism.standard[i-1][2],Algorism.standard[i-1][3],label=f"{i}'s Centroids",marker='x')#중심점
plt.legend()
plt.title(f'results',size = 15)
plt.xlabel('Annual Income',size = 12)
plt.ylabel('Spending Score',size = 12)
plt.show()