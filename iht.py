import numpy as np
import scipy.io
import types

def AIHT(x, A, AT, m, M, thresh):
	"""
	Accelerated iterative Hard thresholding algorithm that keeps exactly M elements 
	in each iteration. This algorithm includes an additional double
	overrelaxation step that significantly improves convergence speed without
	destroiing any of the theoretical guarantees of the IHT algorithm
	detrived in [1], [2] and [3].
	
	This algorithm is used to solve the problem A*z=x
	
	Inputs:
	 x: observation vector to be decomposed
	 A: it can be a (nxm) matrix that gives the effect of the forward matrix A on a vector or an operator that does the same
	 AT: it can be a (nxm) matrix that gives the effect of the backward matrix A.T on a vector or an operator that does the same
	 m: length of the solution vector s
	 M: number of non-zero elements to keep in each iteration
	 thresh: stopping criterion
	 
	Outputs:
	 s: solution vector
	 err_mse: vector containing mse of approximation error for each iteration
	"""
	
	n1, n2 = x.shape
	if (n2 == 1):
		n = n1
	elif (n1 == 1):
		x = x.T
		n = n2
	else:
		exit('x must be a vector')
	
	sigsize = np.dot(x.T, x) / n
	oldERR      = sigsize
	err_mse     = []
	iter_time   = []
	STOPTOL     = 1e-16
	MAXITER     = n**2
	verbose     = True
	initial_given=0
	s_initial   = np.zeros((m,1))
	MU          = 0
	acceleration= 0	
	Count = 0
	
# Define the appropriate functions whether the forward/backward operator is given as a call to a function or a matrix
# This makes everything transparent in the following
	if (isinstance(A, types.FunctionType)):
		P = lambda z: A(z)
		PT = lambda z: AT(z)
	else:
		P = lambda z: np.dot(A, z)
		PT = lambda z: np.dot(AT,z)
	
	s_initial = np.zeros((m,1))
	Residual = x
	s = np.copy(s_initial)
	Ps = np.zeros((n,1))
	oldErr = sigsize
	
	x_test = np.random.randn(m,1)
	x_test = x_test / np.linalg.norm(x_test)
	nP = np.linalg.norm(P(x_test))
	if (np.abs(MU*nP) > 1):
		exit('WARNING! Algorithm likely to become unstable. Use smaller step-size or || P ||_2 < 1.')
		
# Main algorithm
	t = 0
	done = False
	iteration = 1
	min_mu = 1e5
	max_mu = 0
	
	while (not done):
		Count += 1
		if (MU == 0):
			
# Calculate optimal step size and do line search
			if ((Count > 1) & (acceleration == 0)):
				s_very_old = s_old
			s_old = s
			IND = s != 0
			d = PT(Residual)

# If the current vector is zero, we take the largest element in d
			if (np.sum(IND) == 0):
				sortind = np.argsort(np.abs(d), axis=0)[::-1]
				IND[sortind[0:M]] = 1
			
			id = IND * d
			Pd = P(id)
			mu = np.dot(id.T, id) / np.dot(Pd.T, Pd)
			max_mu = np.max([mu,max_mu])
			min_mu = np.min([mu,min_mu])
			mu = min_mu
			s = s_old + mu*d
			sortind = np.argsort(np.abs(s), axis=0)[::-1]
			s[sortind[M:]] = 0
			
			if ((Count > 1) & (acceleration == 0)):
				very_old_Ps = old_Ps
			old_Ps = Ps
			Ps = P(s)
			Residual = x-Ps
						
			if ((Count > 2) & (acceleration == 0)):
# First overrelaxation				
				Dif = (Ps-old_Ps)
				a1 = np.dot(Dif.T, Residual) / np.dot(Dif.T, Dif)
				z1 = s + a1 * (s-s_old)
				Pz1 = (1+a1)*Ps - a1*old_Ps
				Residual_z1 = x-Pz1								
				
				
				
# Second overrelaxation
				Dif = Pz1 - very_old_Ps
				a2 = np.dot(Dif.T, Residual_z1) / np.dot(Dif.T, Dif)
				z2 = z1 + a2 * (z1-s_very_old)
				
# Threshold z2
				sortind = np.argsort(np.abs(z2), axis=0)[::-1]
				z2[sortind[M:]] = 0
				Pz2 = P(z2)
				Residual_z2 = x - Pz2
								

# Decide if z2 is any good
				if (np.dot(Residual_z2.T, Residual_z2) / np.dot(Residual.T, Residual) < 1):
					s = z2
					Residual = Residual_z2
					Ps = Pz2
			
			#if (acceleration > 0):
				#s, Residual = mySubsetCG(x, s, P, Pt
			
# Calculate step-size requirements
			omega = (np.linalg.norm(s-s_old) / np.linalg.norm(Ps-old_Ps))**2
			
						
# As long as the support changes and mu > omega, we decrease mu
			while ((mu > 1.5*omega) & (np.sum(np.logical_xor(IND, s != 0)) != 0) & (np.sum(IND) != 0)):
				print "Decreasing mu"
				
# We use a simple line search, halving mu in each step
				mu = mu / 2
				s = s_old + mu*d
				sortind = np.argsort(np.abs(s), axis=0)[::-1]
				s[sortind[M:]] = 0
				Ps = P(s)
				
# Calculate optimal step size and do line search
				Residual = x - Ps
				if ((Count > 2) & (acceleration == 0)):

# First overrelaxation				
					Dif = (Ps-old_Ps)
					a1 = np.dot(Dif.T, Residual) / np.dot(Dif.T, Dif)
					z1 = s + a1 * (s-s_old)
					Pz1 = (1+a1)*Ps - a1*old_Ps
					Residual_z1 = x-Pz1				
				
# Second overrelaxation
					Dif = Pz1 - very_old_Ps
					a2 = np.dot(Dif.T, Residual_z1) / np.dot(Dif.T, Dif)
					z2 = z1 + a2 * (z1-s_very_old)
					
# Threshold z2
					sortind = np.argsort(np.abs(z2), axis=0)[::-1]
					z2[sortind[M:]] = 0
					Pz2 = P(z2)
					Residual_z2 = x - Pz2
				
# Decide if z2 is any good
					if (np.dot(Residual_z2.T, Residual_z2) / np.dot(Residual.T, Residual) < 1):
						s = z2
						Residual = Residual_z2
						Ps = Pz2
						
# Calculate step-size requirements
				omega = (np.linalg.norm(s-s_old) / np.linalg.norm(Ps-old_Ps))**2
			
			ERR = np.dot(Residual.T, Residual) / n
			err_mse.append(ERR)
			
# Are we done yet?
			gap = np.linalg.norm(s-s_old)**2 / m
			if (gap < thresh):
				done = True
				
			if (not done):
				iteration += 1
				oldERR = ERR
			if (verbose):
				print "Iter={0} - gap={1} - target={2}".format(Count,gap,thresh)
	return s, err_mse