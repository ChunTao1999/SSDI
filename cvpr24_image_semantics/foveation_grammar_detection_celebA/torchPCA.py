import torch
import numpy as np
import matplotlib.pyplot as plt
import pdb

class PCA:
	def __init__(self, Data):
		self.Data = Data

	def __repr__(self):
		return f'PCA({self.Data})'

	@staticmethod
	def Center(Data):
		#Convert to torch Tensor and keep the number of rows and columns
#        t = torch.from_numpy(Data)
		t = Data
#		pdb.set_trace()
		no_rows, no_columns = t.size()
		row_means = torch.mean(t, 1).unsqueeze(1) # (1920, 1)
		#Expand the matrix in order to have the same shape as X and substract, to center
		for_subtraction = row_means.expand(no_rows, no_columns)
		X = t - for_subtraction #centered
		return(X)

	@classmethod
	def Decomposition(cls, Data, k):
		#Center the Data using the static method within the class
		X = cls.Center(Data)
		U,S,V = torch.svd(X)
		eigvecs=U.t()[:,:k] #the first k vectors will be kept
		y=torch.mm(U,eigvecs)
		#Save variables to the class object, the eigenpair and the centered data
		cls.eigenpair = (eigvecs, S)
		cls.data=X
		return(y)

	def explained_variance():
		#Total sum of eigenvalues (total variance explained)
		tot = sum(PCA.eigenpair[1].cpu())
		#Variance explained by each principal component
		var_exp = [(i / tot) for i in sorted(PCA.eigenpair[1].cpu(), reverse=True)]    	
		cum_var_exp = np.cumsum(var_exp)
		#X is the centered data
		X = PCA.data
		#Plot both the individual variance explained and the cumulative:
		plt.bar(range(X.size()[1]), var_exp, alpha=0.5, align='center', label='individual explained variance')
		plt.savefig('./pca_variance_individual.png')
		plt.step(range(X.size()[1]), cum_var_exp, where='mid', label='cumulative explained variance')
		plt.ylabel('Explained variance ratio')
		plt.xlabel('Principal components')
		plt.legend(loc='best')
		plt.savefig('./pca_variance.png')