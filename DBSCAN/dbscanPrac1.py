from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import mglearn
import time


print("DBSCAN start")
datastart = time.time()
print("dataset making")
k = True
X, y = make_moons(n_samples=2000, noise=0.07, random_state=0)
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
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
	#DBSCAN実装
	dbscan = DBSCAN(eps = eps,min_samples=minPts)
	cluster = dbscan.fit_predict(X_scaled)
	end= time.time()

	print("run time",end=":")
	print(end-start)

	#可視化部分
	plt.scatter(X_scaled[:, 0], X_scaled[:,1], c=cluster, cmap=mglearn.cm2, s=60)
	plt.xlabel("Feature 0")
	plt.ylabel("Feature 1")

	plt.show()
	
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

