import sys
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


class NNprac1(nn.Module):
	def __init__(self,inputs=28*28,outputs=10):
		super(NNprac1,self).__init__()
		self.fc1 = nn.Linear(inputs,1000)
		self.fc2 = nn.Linear(1000,outputs)
	def forward(self,x):
		#x = x.view(-1,200)
		x = F.relu(x)
		x = self.fc1(x)
		x = self.fc2(x)

		return F.softmax(x,dim=1)

def data_loader(batch=128,intensity=1.0):
	train_loader = DataLoader(
		datasets.MNIST('./data',
			train=True,
			download=True,
			transform=transforms.Compose([
				transforms.ToTensor(),
				transforms.Lambda(lambda x: x * intensity)
			])
		),
		batch_size=batch,
		shuffle=True
	)

	test_loader = DataLoader(
		datasets.MNIST('./data',
			train=False,
			transform=transforms.Compose([
				transforms.ToTensor(),
				transforms.Lambda(lambda x: x * intensity)
			])
		),
		batch_size = batch,
		shuffle=True
	)

	return train_loader,test_loader

def train(train_data):
	model.train(True)
	print("train start")
	count = 0
	loss = None
	#print("training now:",end="")

	for (data,target) in train_data:
		#print(data.shape)
		#print(target.shape)
		data = data.reshape(-1,28*28)
		optimizer.zero_grad()
		output	= model(data)
		#print("data:",data.shape)
		#print("target:",target.shape)
		#print("output:",output.shape)
		loss = F.nll_loss(output,target)
		loss.backward()
		#print(output)
		#CEL = nn.CrossEntropyLoss()
		#loss = CEL(output,target)
		#loss.backward()
		optimizer.step()
		if count%10 == 0:
			#sys.stdout.write("\rtraining now:%d" % count)
			sys.stdout.write("#")
			sys.stdout.flush()
		count = count + 1
	print()
	return loss

def test(test_data):
	print("test start")
	model.eval()
	loss = 0
	count = 0
	te_data = np.array([])
	te_label = np.array([])

	with torch.no_grad():
		for (data,target) in test_data:
			data = data.reshape(-1,28*28)
			optimizer.zero_grad()
			output	= model(data)
			loss += F.nll_loss(output,target,reduction='sum').item()
			#print(output.shape)
			#print(data.shape)
			te_label = np.append(te_label,output.detach().numpy())
			te_data = np.append(te_data,data)
			if count%10 == 0:
				#sys.stdout.write("\rtraining now:%d" % count)
				sys.stdout.write(">")
				sys.stdout.flush()
			count = count + 1
	print("\ncount:",count)
	loss = loss / count
	

	return loss,te_label,te_data

if __name__ == '__main__':

	#モデルのインスタンス作成
	#model = NNprac1(inputs=3584)
	#model = NNprac1(inputs=28*28)#サンプルコードモデル
	model = NNprac1(inputs=28*28,outputs=10)#CEloss用NN
	cuda = False
	epo = 3
	optimizer = Adam(params=model.parameters(),lr=0.001)
	best_train_loss = np.inf
	tsne = TSNE(n_components=2,random_state=0)
	label_tsne = TSNE(n_components=1,random_state=0)


	if cuda :
		model.cuda()
	print(model)
	#最適化関数設定


	data_tr , data_te = data_loader()
	for Epoch in range(epo):
		print("Epoch:",Epoch)

		loss_tr = train(data_tr)
		if best_train_loss > loss_tr:best_train_loss = loss_tr
		print("train_loss",loss_tr)


	loss_te,te_label,te_data = test(data_te)
	print("best train loss:",best_train_loss,"\ntest loss:",loss_te)
	print("te_label shape:",te_label.shape)
	print("te_data shape :",te_data.shape)

	"""
	te_label = te_label.reshape(10000,10)
	te_data = te_data.reshape(10000,784)
	print("te_label reshape:",te_label.shape)
	print("te_data reshape :",te_data.shape)
	results = tsne.fit_transform(te_data)
	label_results = label_tsne.fit_transform(te_label)	
	print(results.shape)
	print(results[0,:].shape)
	print(results[1,:].shape)
	print(results[:,0].shape)
	print(results[:,1].shape)
	plt.scatter(results[:,0],results[:,1],c=label_results,edgecolor="k")
	plt.show()
	"""








