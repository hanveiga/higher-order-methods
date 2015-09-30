import numpy as np
import copy
import make_movie as mov
from scipy.special import gamma
from scipy.sparse import csc_matrix, diags, find
import sys
import cPickle as pickle


class Conditions(object):
    """ Class for initial and boundary conditions, basic example: sin(period*x) """
    def __init__(self, x):
        self.x = x

    def ic(self, period=np.pi):
        return np.sin(period*self.x)

    def bc(self, a, t, period=np.pi):
        return -np.sin(a*period*t)

class RiemannProblem(Conditions):
    def ic(self):
        flatx = np.squeeze(self.x.flatten(order='F').tolist())
        xmid = 0.5*(flatx[-1]-flatx[0])

        u0 = copy.deepcopy(self.x)

        for i in range(self.x.shape[0]):
            for j in range(self.x.shape[1]):
                if self.x[i,j] < xmid:
                    u0[i,j] = 2
                else:
                    u0[i,j] = 1
      
        return u0

    def bc(self,a, t):
        return 2

def RGKT_coeffs():
    """ Runge kutta coeffs for 4th order time integration """

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
    """ Connectivity and boundary tables for nodes given in the K #
    #% of elements, each with N+1 degrees of freedom.
    #Globals1D;
    #% number volume nodes consecutively
    """
    nodeids = np.reshape(range(0,K*Np), (Np, K), order='F')
    vmapM = np.zeros([Nfp, Nfaces, K])
    vmapP = np.zeros([Nfp, Nfaces, K])
    for k1 in range(0,K):
        for f1 in range(0,Nfaces):
            for i in range(Nfp):
            #find index of face nodes with respect to volume node ordering
                vmapM[i,f1,k1] = nodeids[Fmask[f1],k1]

    for k1 in range(0,K):
        for f1 in range(0,Nfaces):
            #find neighbor
            k2 = EToE[k1,f1]
            f2 = EToF[k1,f1]
            vidM = vmapM[:,f1,k1]
            vidP = vmapM[:,f2,k2]

            x1 = x.flatten(order='F')[0,int(vidM)]
            x2 = x.flatten(order='F')[0,int(vidP)]

            D = (x1-x2)**2
            if (D<NODETOL):
                vmapP[:,f1,k1]= vidP    

    vmapP = vmapP.flatten(order='F')
    vmapM = vmapM.flatten(order='F')

    # Create list of boundary nodes
    mapB = np.where(vmapP==vmapM)
    vmapB = vmapM[mapB]
    
    # Create specific left (inflow) and right (outflow) maps
    mapI = 0
    mapO = K*Nfaces-1
    vmapI = 0
    vmapO = K*Np-1

    return vmapM, vmapP, vmapB, mapB, vmapI, vmapO, mapI, mapO



def JacobiGL(alpha, beta, N):
    """ Compute the N'th order Gauss Lobatto quadrature
    points, x, associated with the Jacobi polynomial,
    of type (alpha,beta) > -1 ( <> -0.5)."""

    x = np.zeros([N+1,1])
    if N==1:
        x[0]=-1.0;
        x[1]=1.0;
        return x

    xint = JacobiGQ(alpha+1,beta+1,N-2)

    xint.sort()
    x[0] = -1
    for i in np.arange(0,len(xint)):
        #print x[i]
        x[i+1] = xint[i]
    x[len(x)-1] = 1

    
    return x

def JacobiGQ(alpha, beta, N):
    """ Compute the N th order Gauss quadrature points, x,
    and weights, w, associated with the Jacobi
    polynomial, of type (alpha,beta) > -1 ( <> -0.5)."""

    x = []
    w = []

    if N==0:
        x[0]=-(alpha-beta)/(alpha+beta+2)
        w[0] = 2
        return x, w

    # Form symmetric matrix from recurrence.
    J = np.zeros(N+1);
    h1 = [2*indx + alpha+beta for indx in range(0,N+1)]
    #print h1

    J = np.diag([-1/2.*(alpha**2-beta**2)/float((h11+2))/float(h11) for h11 in h1]) + \
        np.diag([2./(h1[indx]+2)*np.sqrt((indx+1)*(indx+1 + alpha+ beta)*(indx+1+alpha)*(indx+1+beta)/float(h1[indx]+1)/float(h1[indx]+3)) for indx in np.arange(0,N)],1)

    J = J + np.transpose(J)
    #print J
    [W,V] = np.linalg.eig(J)
    x = W
    #print V[0]
    return x #, w

def Dmatrix1D(N,r,V):
    """ Initialize the (r) differentiation matrices
    % on the interval, evaluated at (r) at order N """

    Vr = GradVandermonde1D(N, r)
    Dr = np.dot(Vr,np.linalg.inv(V))
    #print 'Dr'
    return Dr

def GradVandermonde1D(N,r):
    """Initialize the gradient of the modal basis (i) at (r)
    % at order N"""
    DVr = np.zeros([len(r),N+1])

    for i in range(0,N+1):
        temp = GradJacobiP(r,0,0,i)
        #print temp
        DVr[:,i] = np.transpose(temp)

    return DVr

def GradJacobiP(r, alpha, beta, N):
    """ Evaluate the derivative of the Jacobi polynomial of type
    % (alpha,beta)>-1, at points r for order N and returns
    % dP[1:length(r))] """
    
    dP = np.zeros([len(r), 1])
    if(N == 0):
        dP.fill(0.0)
    else:
        dP = np.reshape(np.sqrt(N*(N+alpha+beta+1))*JacobiP(r,alpha+1,beta+1, N-1),[len(r), 1])
    
    return dP

def Lift1D(V, Np, Nfaces, Nfp):
    """ Compute surface integral term in DG formulation """

    Emat = np.zeros([Np,Nfaces*Nfp])
    Emat[0,0] = 1.0
    Emat[Np-1,1] = 1.0
    
    LIFT = np.dot(V,np.dot(np.transpose(V),Emat))

    return LIFT


def Normals1D(Nfp, Nfaces, K):
    """Compute outward pointing normals at elements faces"""
    nx = np.zeros([Nfp*Nfaces, K]);

    nx[0, :] = -1.0; nx[1, :] = 1.0;

    return nx

def GeometricFactors1D(x, Dr):
    """ Compute the metric elements for the local mappings
    % of the 1D elements"""
    
    xr = Dr*x
    J = xr
    rx = 1/J
    return rx, J

def connect1D(EToV, Nfaces):
    #   function [EToE, EToF] = Connect1D(EToV)
    """ Build global connectivity arrays for 1D grid based
    on standard EToV input array from grid generator"""

    # Find number of elements and vertices
    K = EToV.shape[0]
    TotalFaces = Nfaces*K;
    Nv = K+1;
    
    # List of local face to local vertex connections
    vn = [0,1];
    
    # Build global face to node sparse array
    SpFToV = csc_matrix((TotalFaces, Nv)) #elements*faces, vertices

    sk = 0;

    for k in range(0,K): # number of elements
        for face in range(0,Nfaces): # number of faces
            SpFToV[sk, EToV[k, face]] = 1;
            sk = sk+1;

    SpFToF = SpFToV*np.transpose(SpFToV) - diags([1]*TotalFaces,0,shape=(TotalFaces,TotalFaces)) #np.diags(TotalFaces);

    faces1, faces2, vals = find(SpFToF==1)

    faces1 = [f for f,i in zip(faces1, range(0,len(vals))) if vals[i] == 1]
    faces2 = [f for f,i in zip(faces2, range(0,len(vals))) if vals[i] == 1]

    # truly distasteful way of implementing the FToF conversion to EToE and EToF
    element1 = [np.floor( (elem)/Nfaces ) for elem in faces1]
    face1 = [np.mod( (elem), Nfaces ) for elem in faces1]
    element2 = [ np.floor( (elem)/Nfaces ) for elem in faces2]
    face2 = [ np.mod( (elem), Nfaces ) for elem in faces2]

    indices = []

    for i,j in zip(element1,face1):
        indices.append(np.ravel_multi_index((int(i), int(j)), dims=(K, Nfaces), order='F'))
    EToE = np.transpose(np.tile(range(0,K),(Nfaces,1))).flatten()
    EToF = np.tile(range(0,Nfaces),(K,1)).flatten()

    EToE[indices] = element2
    EToF[indices] = face2

    EToE = np.reshape(EToE, (K,Nfaces), order='F')
    EToF = np.reshape(EToF, (K,Nfaces), order='F')

    return EToE, EToF

def Vandermonde1D(N,r):
    """ Initialize the 1D Vandermonde Matrix, V_{ij} = phi_j(r_i);"""
    V1D = np.zeros([len(r),N+1])
    for j in range(1,N+2):
        V1D[:,j-1]= np.reshape(JacobiP(r, 0, 0, j-1),[len(r)])

    return V1D

def JacobiP(x,alpha,beta,N):
    """ Evaluate Jacobi Polynomial of type (alpha,beta) > -1
    (alpha+beta <> -1) at points x for order N and returns
    P[1:length(xp))]
    Note : They are normalized to be orthonormal.
    Turn points into row if needed. """

    xp = x
    if xp.shape[0] == 1:
        xp = np.transpose(xp)
        return xp

    PL = np.zeros([N+1, len(xp)])


    # Initial values P_0(x) and P_1(x)
    gamma0 = 2**(alpha+beta+1)/float((alpha+beta+1)*gamma(alpha+1)*gamma(beta+1))/float(gamma(alpha+beta+1))
    PL[0,:] = 1.0/float(np.sqrt(gamma0))

    if N==0:
        P = np.transpose(PL)
        return P

    gamma1 = (alpha+1)*(beta+1)/float((alpha+beta+3))*gamma0
    PL[1,:] = np.reshape(((alpha+beta+2)*xp/2. + (alpha-beta)/2.)/float(np.sqrt(gamma1)),[1,len(xp)])
    if N==1:
        P=np.transpose(PL[N,:])
        return P

    # Repeat value in recurrence.
    aold = 2/float(2+alpha+beta)*np.sqrt((alpha+1)*(beta+1)/float(alpha+beta+3))

    for i in range(0,N-1):
        h1 = 2*(i+1)+alpha+beta
        anew = 2./float(h1+2)*np.sqrt( ((i+1)+1)*(i+1+1+alpha + beta)*(i+1+1+alpha) * (i+1+1+beta)/float(h1+1)/float(h1+3))
        #print aold
        bnew = - (alpha**2-beta**2)/float(h1)/float(h1+2)
        temporary= np.multiply((np.transpose(xp)-bnew),PL[i+1,:])
        PL[i+2,:] = 1/float(anew)*( -aold*PL[i,:] + temporary )
        aold = anew
    P = np.transpose(PL[PL.shape[0]-1,:])
    
    return P

class advectionClass(object):
    def __init__(self, N, xintervals):
        self.N = N
        self.Nfp = 1
        self.Np = self.N+1

        self.Nv, self.VX, self.K, self.EToV = generate_mesh(0.0, 2.0, xintervals)
        self.va = self.EToV[:,0]
        self.vb = self.EToV[:,1]

        self.Nfaces = 2

        self.r = JacobiGL(0,0,self.N)
        tempa = [self.VX[int(a)] for a in self.va]
        tempb = [self.VX[int(b)] for b in self.vb]
        self.x = np.dot(np.ones([self.N + 1,1]) , np.matrix(tempa)) + \
            0.5*np.dot(self.r+1,np.matrix(tempb)-np.matrix(tempa))

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

        self.Fscale = 1./np.squeeze(self.J[self.Fmask,:])

        self.EToE, self.EToF = connect1D(self.EToV, self.Nfaces)

        self.vmapM, self.vmapP, self.vmapB, self.mapB, self.vmapI, self.vmapO, self.mapI, self.mapO = \
        BuildMaps1D(self.K, self.Np, self.Nfaces, self.Nfp, self.Fmask, self.EToE, self.EToF, self.x, self.NODETOL)

        self.rgkt = RGKT_coeffs() #dict containing the rk coeffs

        self.conditions=Conditions(self.x)

    def AdvecRHS1D(self, u,time, a):
        """ Purpose : Evaluate RHS flux in 1D advection """

        # form field differences at faces
        alpha=0
        du = np.zeros([self.Nfp*self.Nfaces,self.K])

        tempu_m = [u1 for indx, u1 in zip(range(len(u)),u) if indx in self.vmapM]
        tempu_p = [u1 for indx, u1 in zip(range(len(u)),u) if indx in self.vmapP]

        flatu = np.squeeze(u.flatten(order='F'))
        tempu_m = []
        tempu_p = []
        for indx in self.vmapM:
            tempu_m.append(flatu[0,indx])
        for indx in self.vmapP:
            tempu_p.append(flatu[0,indx])

        du = du.flatten(order='F')
        du[:] = np.multiply(np.array(tempu_m) - np.array(tempu_p),a*self.nx.flatten(order='F') - ((1-alpha)*abs(a*self.nx.flatten(order='F')))/2.)
        #print du
        #uin = -np.sin(a*np.pi*time)
        #uin = 2
        uin = self.conditions.bc(a,time)

        du[self.mapI] = np.dot(u[self.vmapI,0]-uin,(a*self.nx[self.mapI,0])/2.)
        #du[self.mapI] = np.dot(u[self.vmapI,0],(a*self.nx[self.mapI,0])/2.)
        du[self.mapO] = 0
        #print du
        du2 = np.reshape(du, [self.Nfp*self.Nfaces,self.K], order='F')
        rhsu = -a*np.multiply(self.rx,np.dot(self.Dr , u)) + np.dot(self.LIFT,np.multiply(self.Fscale,np.array(du2)))
        #print rhsu
        return rhsu

    def Advec1D(self, FinalTime):
        """ Integrate 1D advection until FinalTime starting with
        initial the condition, u """

        time = 0
       
        #u0 = InitialCondition(self.x).get_riemann()
        u0 = self.conditions.ic()
        #Runge-Kutta residual storage
        resu = np.zeros([self.Np, self.K])
        # compute time step size
        xmin = min(abs(self.x[0,:]-self.x[1,:]))
        CFL = 0.5

        dt = CFL/(2*np.pi)*np.transpose(xmin)[0][0]
        
        dt = .5*dt
        
        Nsteps = np.ceil(FinalTime/float(dt))
        dt = FinalTime/float(Nsteps)
        
        a = 2*np.pi # advection speed
        
        frames = []
        times = []
        step = 0
        for tstep in range(0, int(Nsteps)):
            for INTRK in range(0,5):
                # Runge kutta time integration 4 stage 4th order
                timelocal = time + self.rgkt['rk4c'][INTRK]*dt
                rhsu = self.AdvecRHS1D(u0, timelocal, a)
                resu = self.rgkt['rk4a'][INTRK]*resu + dt*rhsu
                u0 = u0 + self.rgkt['rk4b'][INTRK]*resu
            times.append(time)
            time = time+dt
            frames.append(u0.flatten()[0])
            #break

        #return lol

        mov.make_movie(frames, self.x.flatten()[0], times, "dg"+str(self.N))
        return u0

    def get_l2_error(self,finaltime):

        u0 = self.Advec1D(finaltime)
        #print u0
        flat_x = np.squeeze(self.x.flatten(order='F')).tolist()
        flat_u = np.squeeze(u0.flatten(order='F')).tolist()
        N_x = len(flat_x)
        print N_x
        errortotal = 0
        analytical_solution = lambda x: np.sin(np.pi*(x - 2*np.pi*finaltime))

        a = flat_x[0][0:-2]
        b = flat_x[0][1:-1]
        print b
        print zip(a,b,range(len(a)))

        max_error = 0
        for x1, x2, indx in zip(a,b,range(len(a))):
            error_min = error(analytical_solution, flat_u[0][indx], x1)
            error_max = error(analytical_solution, flat_u[0][indx+1], x2)
            errortotal =+ quad_rule(x1,x2,error_max**2,error_min**2)

        return np.sqrt(errortotal)

    def get_max_error(self, finaltime):
        u0 = self.Advec1D(finaltime)
        
        anasolution = np.sin(np.pi*(self.x - 2*np.pi*finaltime))
        
        return np.max(np.abs(u0-anasolution))

def error(analytical_solution, approx_solution, x):
    error_e = np.abs(analytical_solution(x)-approx_solution)
    return error_e

def quad_rule(xmin, xmax, fc_max, fc_min):
    evaluated = (xmax-xmin)/2. * (fc_max+fc_min)
    return evaluated

def gaussian_quad_rule():
    evaluated = 0
    return evaluated

def generate_mesh(xmin, xmax, K):
    """ Generate simple equidistant grid with K elements """
    Nv = K+1
    # Generate node coordinates
    
    VX = range(0,Nv)
    
    for i in VX:
        VX[i] = (xmax-xmin)*(i+1-1)/float((Nv-1)) + xmin

    # read element to node connectivity
    EToV = np.zeros([K, 2])
    
    for k in range(0,K):
        EToV[k,0] = k
        EToV[k,1] = k+1

    return Nv, VX, K, EToV

def run(order, totaltime=1, xintervals=10):
    N = order #order of basis polynomials
    eq = advectionClass(N, xintervals)
    finaltime=totaltime
    u = eq.Advec1D(finaltime)

if __name__ == '__main__':
    run(int(sys.argv[1]))