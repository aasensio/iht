import numpy as np
import iht

def forward(z):	
	return np.dot(A,z)

def backward(z):
	return np.dot(A.T,z)

#out = scipy.io.loadmat('/scratch/caca/sparsify_0_5/HardLab/variables.mat')
#A = out['A']
#x = out['x']
#sol = np.dot(A, x)

x = np.zeros((20,1))
ind = np.random.permutation(20)
x[ind[0:3]] = np.random.randn(3,1)
A = np.random.randn(15,20)
sol = np.dot(A,x)

mLength = 20
mSparsity = 3

## Using matrices
res, err = iht.AIHT(sol, A, A.T, mLength, mSparsity, 1e-8)

## Using operators
#res, err = iht.AIHT(sol, forward, backward, mLength, mSparsity, 1e-8)