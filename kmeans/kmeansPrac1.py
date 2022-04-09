import numpy as np
from sklearn.datasets import make_moons,make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans,DBSCAN
import matplotlib.pyplot as plt
import time

def MakePlot(plt,x,y,c,labelx,labely,title="TITLE"):
	#表示
	plt.scatter(x, y, c=c,s=25,edgecolor="k")
	plt.xlabel(labelx)
	plt.ylabel(labely)
	plt.title(title)
	plt.savefig(title+'_kmeans.png')
	plt.tight_layout()
	plt.show()
	plt.clf()

#ブロブデータセット
cluster = 6#クラスタ数
cluster_std = 1.2#分散
features = 120#次元数

#DBSCANのパラメータ
#eps = 5
#minPts = 10

bX,Y = make_blobs(random_state=10,n_samples=2000,n_features=features,cluster_std=cluster_std,centers=cluster)
scaler2 = StandardScaler()
scaler2.fit(bX)
bX_scaled = scaler2.transform(bX)

start = time.time()

kmcluss = KMeans(n_clusters=cluster).fit_predict(bX)
#dbcluss = DBSCAN(eps=eps,min_samples=minPts).fit_predict(bX)

kmcluss_std = KMeans(n_clusters=cluster).fit_predict(bX_scaled)
#dbcluss2_std = DBSCAN(eps=eps-0.43,min_samples=minPts).fit_predict(bX_scaled)

end = time.time()

TIME = end - start

#print(dbcluss)

#print(Y)
#print(kmcluss)
print("time :"+(str)(TIME))

#正解作成
MakePlot(plt,bX[:, 0], bX[:,1],Y,"Feature 0","Feature 1","anser")
MakePlot(plt,bX_scaled[:, 0], bX_scaled[:,1],Y,"Feature 0","Feature 1","anser_std")
#blob可視化部分
MakePlot(plt,bX[:, 0], bX[:,1],kmcluss,"Feature 0","Feature 1","kmeans")
#MakePlot(plt,bX[:, 0], bX[:,1],dbcluss,"Feature 0","Feature 1","DBSCAN")

MakePlot(plt,bX_scaled[:, 0], bX_scaled[:,1],kmcluss_std,"Feature 0","Feature 1","kmeans_std")
#MakePlot(plt,bX_scaled[:, 0], bX_scaled[:,1],dbcluss2_std,"Feature 0","Feature 1","DBSCAN_std")


