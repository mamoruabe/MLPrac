from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN,KMeans
import matplotlib.pyplot as plt
#import mglearn
import time , os ,glob

def dircheck(PATH):
	print("ディレクトリ初期化")
	if os.path.exists(PATH):
		for p in glob.glob(PATH+"*.png"):
			os.remove(p)
	else:
		print("該当するディレクトリを作成")
		os.makedirs(PATH)


def MakePlot(x,y,c,labelx,labely,eps=0,title="TITLE",sh = True):
	#表示
	plt.clf()#初期化


	ensstr = str(eps)
	if eps == 0 :
		titlestr = title
	else:
		titlestr = title + "  " +"eps = " + ensstr

	plt.scatter(x, y, c=c,s=25,edgecolor="k")
	plt.xlabel(labelx)
	plt.ylabel(labely)
	plt.title(titlestr)
	#fig = plt.figure()
	plt.savefig(PATH+title+'.png')#画像で保存。上書き不可	
	if sh :plt.show()#表示
	pathList.append(PATH+title+'.png')

	#return PATH+title+'.png',fig

print("DBSCAN start")

print("enter features:",end="")
num = int(input())

#パラメータ
#その他初期設定
k = True
features = num
minPts = 10 #minPtsは変化させない
PATH = './dbscan3result_features_'+str(features)+'/'



datastart = time.time()


print("dataset making")
print("features:",features)
#三日月形のデータセット
mX, my = make_moons(n_samples=2000, noise=0.1, random_state=8)
scaler = StandardScaler()
scaler.fit(mX)
mX_scaled = scaler.transform(mX)

#ブロブデータセット
bX,by = make_blobs(random_state=10,n_samples=2000,n_features=features,cluster_std=1.2,centers=6)
scaler2 = StandardScaler()
scaler2.fit(bX)
bX_scaled = scaler2.transform(bX)

#時間計測
dataend = time.time()
dtime = dataend - datastart
print("finish:"+(str)(dtime))


while(k):
	#初期化ゾーン
	dircheck(PATH) 
	pathList = list()
	#figlist = list()



	print("please enter parameters")
	print("eps_std=",end=":")
	eps_std = (float)(input())

	print("eps_nml=",end=":")
	eps_nml = (int)(input())

	#start = time.time()
	#end= time.time()
	#print("run time",end=":")
	#print(end-start)
	
	#DBSCAN実装
	Mdbscan_std = DBSCAN(eps = eps_std/eps_nml,min_samples=minPts).fit_predict(mX_scaled)
	Mdbscan_nml = DBSCAN(eps = eps_nml/eps_nml-0.9,min_samples=minPts).fit_predict(mX)
	Bdbscan_std = DBSCAN(eps = eps_std,min_samples=minPts).fit_predict(bX_scaled)
	Bdbscan_nml = DBSCAN(eps = eps_nml,min_samples=minPts).fit_predict(bX)

	#Bdbscan_nml_nparam = DBSCAN().fit_predict(bX)
	#kmcl = KMeans().fit_predict(bX)
	#print(Bdbscan_nml_nparam)
	#print(Y)

	#可視化部分
	MakePlot(bX_scaled[:, 0], bX_scaled[:,1], Bdbscan_std,"Feature0","Feature1",eps_std,"dbscan3_std_blob")
	MakePlot(mX_scaled[:, 0], mX_scaled[:,1],Mdbscan_std,"Feature0","Feature1",eps_std,"dbscan3_std_Moon")
	MakePlot(bX[:, 0], bX[:,1], Bdbscan_nml,"Feature0","Feature1",eps_nml,"dbscan3_nml_blob")
	MakePlot(mX[:, 0], mX[:,1], Mdbscan_nml,"Feature0","Feature1",eps_nml,"dbscan3_nml_Moon")
	

	#MakePlot(bX[:, 0], bX[:,1], Bdbscan_nml_nparam,"Feature0","Feature1",10000,"dbscan3_nml_nparam")
	#pathList.append(MakePlot(bX[:, 0], bX[:,1], kmcl,"Feature0","Feature1",10000,"kmeans_nparam"))

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
		else:
			print("please enter y or n")

MakePlot(bX[:, 0], bX[:,1], by,"Feature0","Feature1",title="dbscan3_nml_anser_blob",sh = False)
MakePlot(mX[:, 0], mX[:,1], my,"Feature0","Feature1",title="dbscan3_nml_anser_moon",sh = False)


print("保存したファイルリスト")
for ph in pathList:
	print(ph)

print("See you")


"""
pathList.append(k)
figlist.append(p)
for pl in figlist:
	print(pl)
print('保存中')
for i in range(len(figlist)):
	print(i)
	figlist[i].savefig(pathList[i])#画像で保存。上書き不可
"""
