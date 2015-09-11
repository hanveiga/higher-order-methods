import numpy as np
import copy
import make_movie as mov
from scipy.special import gamma
from scipy.sparse import csc_matrix, diags, find

def RGKT_coeffs():
	# ew 
	rgkt = {}
	rgkt['rk4a'] = [ 0.0, \
					 -567301805773.0/1357537059087.0, \
					-2404267990393.0/2016746695238.0, \
					-3550918686646.0/2091501179385.0, \
					-1275806237668.0/842570457699.0]

	rgkt['rk4b'] = [ 1432997174477.0/9575080441755.0, \
					5161836677717.0/13612068292357.0, \
					1720146321549.0/2090206949498.0, \
					3134564353537.0/4481467310338.0, \
					2277821191437.0/14882151754819.0]

	rgkt['rk4c'] = [ 0.0, \
					1432997174477.0/9575080441755.0, \
					2526269341429.0/6820363962896.0, \
					2006345519317.0/3224310063776.0, \
					2802321613138.0/2924317926251.0]
	
	return rgkt

def BuildMaps1D(K, Np, Nfaces, Nfp, Fmask, EToE, EToF, x, NODETOL):
	#function [vmapM, vmapP, vmapB, mapB] = BuildMaps1D
	#% function [vmapM, vmapP, vmapB, mapB] = BuildMaps1D
	#% Purpose: Connectivity and boundary tables for nodes given in the K #
	#% of elements, each with N+1 degrees of freedom.
	#Globals1D;
	#% number volume nodes consecutively
	nodeids = np.reshape(range(0,K*Np), (Np, K))

	vmapM = np.zeros([Nfp, Nfaces, K])
	vmapP = np.zeros([Nfp, Nfaces, K])
	for k1 in range(0,K):
		for f1 in range(0,Nfaces):
			#find index of face nodes with respect to volume node ordering
				vmapM[:,f1,k1] = nodeids[Fmask[f1], k1]

	for k1 in range(0,K):
		for f1 in range(0,Nfaces):
			#find neighbor
			try:
				k2 = EToE[k1,f1]
				f2 = EToF[k1,f1]
				# find volume node numbers of left and right nodes
				vidM = vmapM[:,f1,k1]
				vidP = vmapM[:,f2,k2]

				x1 = [x[0][int(vidM)]]
				x2 = [x[0][int(vidP)]]
				# Compute distance matrix
				D = [ (x11-x22)**2  for x11, x22 in zip(x1,x2)] 
				if (D<NODETOL):
					vmapP[:,f1,k1]= vidP	
			except:
				continue

	vmapP = vmapP.flatten()
	vmapM = vmapM.flatten()

	# Create list of boundary nodes
	mapB = np.where(vmapP==vmapM)
	vmapB = vmapM[mapB]
	
	# Create specific left (inflow) and right (outflow) maps
	mapI = 1
	mapO = K*Nfaces
	vmapI = 1
	vmapO = K*Np

	return vmapM, vmapP, vmapB, mapB, mapI, mapO, vmapI, vmapO


def JacobiGL(alpha, beta, N):
	""" Compute the N'th order Gauss Lobatto quadrature
	points, x, associated with the Jacobi polynomial,
	of type (alpha,beta) > -1 ( <> -0.5)."""

	x = np.zeros([N+1,1]);
	if N==1:
		x[0]=-1.0;
		x[1]=1.0;
		return x

	xint , w = JacobiGQ(alpha+1,beta+1,N-1);

	x[0] = -1
	for i in np.arange(0,len(xint)):
		x[i+1] = xint[i-1]
	x[len(x)-1] = 1

	
	return x

def JacobiGQ(alpha, beta, N):
	""" Compute the N th order Gauss quadrature points, x,
	and weights, w, associated with the Jacobi
	polynomial, of type (alpha,beta) > -1 ( <> -0.5)."""

	x = []
	w = []

	if N==0:
		x[0]=(alpha-beta)/(alpha+beta+2)
		w[0] = 2
		return x, w

	# Form symmetric matrix from recurrence.
	J = np.zeros(N+1);
	h1 = [2*indx + alpha+beta for indx in range(0,N)]
	J = np.diag([-1/2.*(alpha^2-beta^2)/float((h11+2))/float(h11) for h11 in h1]) + \
		np.diag([2./float((h11+2))*np.sqrt(h11*(h11+alpha+beta)* (h11+alpha)*(h11+beta)/float((h11+1))/float((h11+3))) for h11 in h1[1:N]],1)

	#diag([2./(h1(1:N)+2).*sqrt((1:N).*((1:N)+alpha+beta).*... #upper diagonal, b_i
	#((1:N)+alpha).*((1:N)+beta)./(h1(1:N)+1)./(h1(1:N)+3)),1)

	#if (alpha+beta<10*eps):
	#	J[1,1]=0.0

	J = J + np.transpose(J)

	[V,D] = np.linalg.eig(J)
	x = np.diag(D)
	w = [((v**2)*2)**(alpha+beta+1)/float((alpha+beta+1))*gamma(alpha+1)*gamma(beta+1)/float(gamma(alpha+beta+1)) for v in V]

	#w = [(V(1,:)'').^2*2^(alpha+beta+1)/(alpha+beta+1)*gamma(alpha+1)*...
	#gamma(beta+1)/gamma(alpha+beta+1) for v in V[1,:]]
	return x, w

def Dmatrix1D(N,r,V):
	""" Initialize the (r) differentiation matrices
	% on the interval, evaluated at (r) at order N """

	Vr = GradVandermonde1D(N, r)
	#Dr = np.divide(Vr,V)
	Dr = np.linalg.lstsq(np.transpose(Vr),np.transpose(V))[0]

	return Dr

def GradVandermonde1D(N,r):
	"""Initialize the gradient of the modal basis (i) at (r)
	% at order N"""
	DVr = np.zeros([len(r),N+1])

	for i in range(0,N-1):
		temp = GradJacobiP(r,0,0,i)
		DVr[:,i+1] = copy.deepcopy(np.transpose(temp)[0]) #first entry of DVr has nothing
	
	return DVr

def GradJacobiP(r, alpha, beta, N):
	""" Evaluate the derivative of the Jacobi polynomial of type
	% (alpha,beta)>-1, at points r for order N and returns
	% dP[1:length(r))] """
	
	dP = np.zeros([len(r), 1])
	if(N == 0):
		dP.fill(0.0)
	else:
		dP = np.sqrt(N*(N+alpha+beta+1))*JacobiP(r,alpha+1,beta+1, N-1);
	
	return dP

def Lift1D(V, Np, Nfaces, Nfp):
	""" Compute surface integral term in DG formulation """

	Emat = np.zeros([Np,Nfaces*Nfp])
	Emat[0,0] = 1.0
	Emat[Np-1,1] = 1.0
	
	LIFT = np.dot(V,np.dot(np.transpose(V),Emat))
	#print LIFT

	return LIFT


def Normals1D(Nfp, Nfaces, K):
	"""Compute outward pointing normals at elements faces"""
	nx = np.zeros([Nfp*Nfaces, K]);

	nx[0, :] = -1.0; nx[1, :] = 1.0;

	return nx

def GeometricFactors1D(x, Dr):
	""" Compute the metric elements for the local mappings
	% of the 1D elements"""
	
	xr = np.transpose(Dr)*x
	J = xr
	rx = [1./j for j in J[0]]
	return rx, J

def connect1D(EToV, Nfaces):
	#	function [EToE, EToF] = Connect1D(EToV)
	""" Build global connectivity arrays for 1D grid based
	on standard EToV input array from grid generator"""

	# Find number of elements and vertices
	K = EToV.shape[1]
	TotalFaces = Nfaces*K;
	Nv = K+1;
	
	#% List of local face to local vertex connections
	vn = [0,1];
	
	#% Build global face to node sparse array
	
	#SpFToV = spalloc(TotalFaces, Nv, 2*TotalFaces);
	
	SpFToV = csc_matrix((TotalFaces, Nv))
	#print 'etov'
	#print EToV
	sk = 0;
	for k in range(0,K): #=1:K
		for face in range(0,Nfaces): #=1:Nfaces:
			SpFToV[sk, EToV[k, vn[face]]] = 1;
			sk = sk+1;

	#% Build global face to global face sparse array
	#print SpFToV
	SpFToF = SpFToV*np.transpose(SpFToV) - diags([1]*TotalFaces,0,shape=(TotalFaces,TotalFaces)) #np.diags(TotalFaces);
	#print SpFToF
	# Find complete face to face connections
	faces1, faces2, vals = find(SpFToF==1)
	faces1 = [f for f,i in zip(faces1, range(0,len(vals))) if vals[i] == 1]
	faces2 = [f for f,i in zip(faces2, range(0,len(vals))) if vals[i] == 1]

	#faces2 = SpFToF[faces1] 
	#print 'made it here'
	#print faces1
	#print faces2
	# Convert face global number to element and face numbers
	#print faces1
	#print faces2
	element1 = [np.floor( (elem-1)/Nfaces ) + 1 for elem in faces1]
	#print element1
	face1 = [np.mod( (elem-1), Nfaces ) + 1 for elem in faces1]
	element2 = [ np.floor( (elem-1)/Nfaces ) + 1 for elem in faces2]
	face2 = [ np.mod( (elem-1), Nfaces ) + 1 for elem in faces2]
	# Rearrange into Nelements x Nfaces sized arrays
	#ind = sub2ind([K, Nfaces], element1, face1);
	#ind = np.ravel_multi_index( [K, Nfaces] , dims=[element1, face1])
	EToE = np.transpose(np.tile(range(0,K),(Nfaces,1)))
	EToF = np.tile(range(0,Nfaces),(K,1))

	#EToE = (1:K)'*ones(1,Nfaces)
	#EToF = ones(K,1)*(1:Nfaces)
	EToE[1] = element2; EToF[1] = face2;
	#print EToE[1]
	#print EToF[1]

	return EToE, EToF

def Vandermonde1D(N,r):
	""" Initialize the 1D Vandermonde Matrix, V_{ij} = phi_j(r_i);"""
	V1D = np.zeros([len(r),N+1])
	for j in range(1,N+1):
		V1D[:,j]= JacobiP(r, 0, 0, j-1) #first element of V1D is empty

	return V1D

def JacobiP(x,alpha,beta,N):
	""" Evaluate Jacobi Polynomial of type (alpha,beta) > -1
	(alpha+beta <> -1) at points x for order N and returns
	P[1:length(xp))]
	Note : They are normalized to be orthonormal.
	Turn points into row if needed. """
	#print' entered jacobiP'
	#print N
	xp = np.transpose(x)[0]

	PL = np.zeros([N+1, len(xp)])


	# Initial values P_0(x) and P_1(x)
	gamma0 = 2**(alpha+beta+1)/float((alpha+beta+1)*gamma(alpha+1)*gamma(beta+1))/float(gamma(alpha+beta+1))
	PL[0,:] = 1.0/float(np.sqrt(gamma0))
	#print 'gamma0 %d', gamma0

	if N==0:
		P = PL
		return P

	gamma1 = (alpha+1)*(beta+1)/float((alpha+beta+3))*gamma0
	PL[1,:] = ((alpha+beta+2)*xp/2. + (alpha-beta)/2.)/float(np.sqrt(gamma1))
	#print PL[0,:]
	if N==1:
		P=PL[N,:]
		return P

	# Repeat value in recurrence.
	aold = 2/(2+alpha+beta)*np.sqrt((alpha+1)*(beta+1)/(alpha+beta+3))
	# Forward recurrence using the symmetry of the recurrence.
	for i in range(0,N-1):
		h1 = 2*(i+1)+alpha+beta
		anew = 2/float((h1+2))*np.sqrt( (i+2)*(i+2+alpha+beta)*(i+2+alpha)*(i+2+beta)/float((h1+1))/float((h1+3)))
		bnew = - (alpha**2-beta**2)/float(h1)/float(h1+2)
		PL[i+1,:] = np.dot(1/float(anew)*( -aold*PL[i,:] + (xp-bnew)), PL[i,:])
		aold =anew
	#print PL
	P = PL[N-1,:]

	#print P

	return P

class advectionClass(object):
	def __init__(self, N):
		self.N = N
		self.Nfp = 1
		self.Np = self.N+1

		self.Nv, self.VX, self.K, self.EToV = generate_mesh(0.0, 2.0, 10)
		self.va = self.EToV[:,0]
		self.vb = self.EToV[:,1]

		self.Nfaces = 2

		self.r = JacobiGL(0,0,self.N)
		print self.r
		tempa = [self.VX[int(a)] for a in self.va]
		tempb = [self.VX[int(b)] for b in self.vb]
		print np.dot(np.ones([self.N + 1,1]), np.matrix(tempa))
		print 0.5*np.dot(self.r+1,np.matrix(tempb)-np.matrix(tempa))
		self.x = np.dot(np.ones([self.N + 1,1]) , np.matrix(tempa)) + \
			0.5*np.dot(self.r+1,np.matrix(tempb)-np.matrix(tempa))

		print 'selfx'
		print self.x

		self.NODETOL = 1e-10

		self.V = Vandermonde1D(N, self.r)

		self.Dr = Dmatrix1D(self.N, self.r, self.V)

		self.rx, self.J = GeometricFactors1D(self.x, self.Dr)

		fmask1 = [indx for indx, val in enumerate(np.abs(self.r+1)) if val < self.NODETOL] # ]np.where( np.abs(self.r + 1) < self.NODETOL)
		fmask2 = [indx for indx, val in enumerate(np.abs(self.r-1)) if val < self.NODETOL] #np.where( np.abs(self.r - 1) < self.NODETOL)
		self.Fmask = [fmask1, fmask2]

		self.LIFT = Lift1D(self.V, self.Np, self.Nfaces, self.Nfp)

		self.nx = Normals1D(self.Nfp, self.Nfaces, self.K)
		indices = [ind for ind in fmask1 + fmask2]

		self.Fx = self.x[self.Fmask,:]
		self.Fscale = np.divide(1.,self.J[self.Fmask,:][0,:,:])

		self.EToE, self.EToF = connect1D(self.EToV, self.Nfaces)

		self.vmapM, self.vmapP, self.vmapB, self.mapB, self.vmapI, self.vmapO, self.mapI, self.mapO = \
		BuildMaps1D(self.K, self.Np, self.Nfaces, self.Nfp, self.Fmask, self.EToE, self.EToF, self.x, self.NODETOL)
		#(K, Np, Nfaces, Nfp, Fmask, EToE, EToF):

		#self.vmapI = []
		#self.vmapO = []
		#self.mapI = []
		#self.mapO = []

		#self.invV= np.linalg.inv(self.V)

		self.rgkt = RGKT_coeffs() #dict

	def AdvecRHS1D(self, u,time, a):
		""" Purpose : Evaluate RHS flux in 1D advection """

		# form field differences at faces
		alpha=1
		du = np.zeros([self.Nfp*self.Nfaces,self.K])

		#print self.vmapM
		tempu_m = [u1 for indx, u1 in zip(range(len(u)),u) for indx in self.vmapM]
		tempu_p = [u1 for indx, u1 in zip(range(len(u)),u) for indx in self.vmapP]
		#print tempu_m[0]
		#print (a*self.nx.flatten()-(1-alpha)*abs(a*self.nx.flatten()))/2.
		#du = (np.array(tempu_m[0])-np.array(tempu_p[0]))*(a*self.nx.flatten()-(1-alpha)*abs(a*self.nx.flatten()))/2.
		# impose boundary condition at x=0
		uin = -np.sin(a*time)
		
		#du [mapI] = (u(vmapI)- uin ).*(a*nx(mapI)-(1-alpha)*abs(a*nx(mapI)))/2;
		#du [mapO] = 0;

		#compute right hand sides of the semi-discrete PDE
		#rhsu = -np.matrix(np.transpose(self.rx))*(np.matrix(np.transpose(self.Dr*u))) + np.matrix(self.LIFT)*(np.matrix(du))

		rhsu = -a*np.multiply(self.rx[0],np.dot(self.Dr , np.array(u))) + np.dot(self.LIFT,np.multiply(self.Fscale,np.array(du)))
		
		return rhsu

	def Advec1D(self, FinalTime):
		""" Integrate 1D advection until FinalTime starting with
		initial the condition, u """

		time = 0
		u0 = np.sin(self.x)
		#Runge-Kutta residual storage
		resu = np.zeros([self.Np, self.K])
		# compute time step size
		xmin = min(abs(self.x[0,:]-self.x[1,:]))#self.x(1,:)-x(2,:)))
		#print np.transpose(xmin)[0][0]
		CFL = 0.75
		
		dt = CFL/(2*np.pi)*np.transpose(xmin)[0][0]
		
		dt = .5*dt
		
		Nsteps = np.ceil(FinalTime/float(dt))
		dt = FinalTime/float(Nsteps)
		# advection speed
		a = 2*np.pi
		
		# outer time step loop

		frames = []
		for tstep in range(0, int(Nsteps)):
			for INTRK in np.arange(1,5):
				timelocal = time + self.rgkt['rk4c'][INTRK]*dt;
				rhsu = self.AdvecRHS1D(u0, timelocal, a);
				resu = self.rgkt['rk4a'][INTRK]*resu + dt*rhsu;
				u0 = u0 + self.rgkt['rk4b'][INTRK]*resu
			#% Increment time
			time = time+dt;
			print u0.shape
			frames.append(u0.flatten()[0,0:9])
		
		mov.make_movie(frames, self.x.flatten()[0,0:9], "dg")

		return u0

def generate_mesh(xmin, xmax, K):
	""" Generate simple equidistant grid with K elements """
	Nv = K+1;
	# Generate node coordinates
	
	VX = range(0,Nv)
	
	for i in VX:
		VX[i] = (xmax-xmin)*(i-1)/float((Nv-1)) + xmin

	# read element to node connectivity
	EToV = np.zeros([K, 2])
	
	for k in range(0,K):
		EToV[k,0] = k
		EToV[k,1] = k+1

	return Nv, VX, K, EToV

def run():

	N = 8 #order of polynomials

	eq = advectionClass(N)

	# initial conditions
	#u0 = np.sin()

	finaltime=0.5

	u = eq.Advec1D(finaltime)

if __name__ == '__main__':
	run()