from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import mglearn
import time

#ループの変数
k = True

print("DBSCAN start")
datastart = time.time()


print("dataset making")
#三日月形のデータセット
mX, y = make_moons(n_samples=8000, noise=0.05, random_state=0)
scaler = StandardScaler()
scaler.fit(mX)
mX_scaled = scaler.transform(mX)

#ブロブデータセット
bX,Y = make_blobs(random_state=8,n_samples=10000,n_features=2,cluster_std=1.2,centers=20)
scaler2 = StandardScaler()
scaler2.fit(bX)
bX_scaled = scaler2.transform(bX)




#時間計測
dataend = time.time()
dtime = dataend - datastart
print("finish:"+(str)(dtime))

while(k):
	print("please enter parameters")
	print("eps=",end=":")
	eps = (float)(input())

	print("minPts=",end=":")
	minPts = (int)(input())

	start = time.time()
	#DBSCAN実装(moon)
	Mdbscan = DBSCAN(eps = eps,min_samples=minPts)
	Mcluster = Mdbscan.fit_predict(mX_scaled)
	end= time.time()

	print("run time",end=":")
	print(end-start)

	
	#moon可視化部分
	plt.scatter(mX_scaled[:, 0], mX_scaled[:,1], c=Mcluster, cmap=mglearn.cm2, s=60)
	plt.xlabel("M Feature 0")
	plt.ylabel("M Feature 1")
	plt.show()
	

	#DBSCAN実装(blob)
	Bdbscan = DBSCAN(eps = eps,min_samples=minPts).fit_predict(bX)
	print(Bdbscan)
	print(Y)

	#blob可視化部分
	plt.scatter(bX_scaled[:, 0], bX_scaled[:,1], c=Bdbscan,s=25,edgecolor="k")
	plt.xlabel("B Feature 0")
	plt.ylabel("B Feature 1")
	plt.show()


	#パラメータ再設定ループ
	print("continue? y or n",end=":")
	
	n = True
	while(n):
		inK = input()
		if inK == "y":
			k = True
			n = False
		elif inK == "n":
			k = False
			n = False
			print("See you")
		else:
			print("please enter y or n")

