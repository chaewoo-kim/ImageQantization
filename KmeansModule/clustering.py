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