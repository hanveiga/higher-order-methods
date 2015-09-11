""" Implementation of WENO method as described in 
"High Order Weighted Essentially Non-Oscillatory Schemes for Convection
Dominated Problems" by Chi Wang Shu.

Time integration: 3 stage RGKT 
Space integration: WENO5 with Lax-Friedrichs Flux splitting
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import make_movie as mov
import scipy as sp

class Flux(object):

	def linear(self,w):
		c=5
		f = lambda x: c*x
		df = lambda x: c
		return f, df

class InitialConditions(object):
	def __init__(self, x):
		self.x = x

	def get_gaussian_IC(self):
		Lx = self.x[len(self.x)-1] - self.x[0]
		xmid = 0.5*(self.x[len(self.x)-1])+self.x[0]

		u0 = np.exp(-20*(self.x-xmid)**2)

		return u0

	def get_sin_IC(self):
		u0 = 1 - np.sin(np.pi * self.x) 
		return u0

def commonIC(x):
	Lx = x(len(x)-1) - x(0)
	xmid = 0.5*(x(len(x)-1)+x(0))

	#u0 = np.sin(np.pi * x)
	u0 = np.exp(-20*(x-xmid)**2)

	return u0

def run():
	Nx = 100
	tEnd = 20
	CFL = 0.2
	xmin = -1
	xmax = 1
	dx = xmax-xmin / float(Nx)

	x = np.linspace(xmin,xmax,Nx)

	u0 = InitialConditions(x).get_gaussian_IC()

	t = 0

	u = u0

	frames = []

	while t < tEnd:
		dt = CFL*dx/max(np.abs(u))-CFL*dx/max(np.abs(u))/2
		if (t+dt > tEnd) or (dt == tEnd-t):
			break

		t=t+dt

		uo = u
		flux , dflux = Flux().linear(u)
		flux = map(flux, u)
		dflux = map(dflux,u)
		S = np.zeros([len(u),1])

		# 1st stage RGKT
		dF = WENO5resAdv1d(u,flux,dflux,S,dx)
		u = uo-dt*dF
		
		flux , dflux = Flux().linear(u)
		flux = map(flux, u)
		dflux = map(dflux,u)
		# 2nd Stage
		dF = WENO5resAdv1d(u,flux,dflux,S,dx)
		u = 0.75*uo+0.25*(u-dt*dF)

		flux , dflux = Flux().linear(u)
		flux = map(flux, u)
		dflux = map(dflux,u)
		# 3rd stage
		dF = WENO5resAdv1d(u,flux,dflux,S,dx)
		u = (uo+2*(u-dt*dF))/3

		frames.append(u)
		
	mov.make_movie(frames,x)

def WENO5resAdv1d(w, flux, dflux, S, dx):

	a=max(np.abs(dflux))
	v=0.5*(flux+a*w)
	u=np.roll(0.5*(flux-a*w),-1)

	vmm = np.roll(v,2)
	vm  =  np.roll(v,1)
	vp  =  np.roll(v,-1)
	vpp =  np.roll(v,-2)

	p0n = (2*vmm - 7*vm + 11*v)/6.
	p1n = ( -vm  + 5*v  + 2*vp)/6.
	p2n = (2*v   + 5*vp - vpp )/6.

	B0n = 13/12.*(vmm-2*vm+v  )**2 + 1/4.*(vmm-4*vm+3*v)**2 
	B1n = 13/12.*(vm -2*v +vp )**2 + 1/4.*(vm-vp)**2
	B2n = 13/12.*(v  -2*vp+vpp)**2 + 1/4.*(3*v-4*vp+vpp)**2

	d0n = 1/10.
	d1n = 6/10.
	d2n = 3/10.
	epsilon = 1e-6

	alpha0n = np.divide(d0n,(epsilon + B0n)**2)
	alpha1n = np.divide(d1n,(epsilon + B1n)**2)
	alpha2n = np.divide(d2n,(epsilon + B2n)**2)
	alphasumn = alpha0n + alpha1n + alpha2n;

	w0n = np.divide(alpha0n,alphasumn)
	w1n = np.divide(alpha1n,alphasumn)
	w2n = np.divide(alpha2n,alphasumn)

	# Numerical Flux at cell boundary, $u_{i+1/2}^{-}$;
	hn = np.multiply(w0n,p0n) + np.multiply(w1n,p1n) + np.multiply(w2n,p2n)

	# Left Flux 
	# Choose the negative fluxes, 'u', to compute the left cell boundary flux:
	#% $u_{i-1/2}^{+}$ 
	umm =  np.roll(u,2)
	um  =  np.roll(u,1)
	up  =  np.roll(u,-1)
	upp =  np.roll(u,-2)

	#% Polynomials
	p0p = ( -umm + 5*um + 2*u  )/6.
	p1p = ( 2*um + 5*u  - up   )/6.
	p2p = (11*u  - 7*up + 2*upp)/6.

	#% Smooth Indicators (Beta factors)
	B0p = 13/12*(umm-2*um+u  )**2 + 1/4*(umm-4*um+3*u)**2
	B1p = 13/12*(um -2*u +up )**2 + 1/4*(um-up)**2
	B2p = 13/12*(u  -2*up+upp)**2 + 1/4*(3*u -4*up+upp)**2

	# Constants
	d0p = 3/10.
	d1p = 6/10.
	d2p = 1/10.
	epsilon = 1e-6

	# Alpha weights 
	alpha0p = np.divide(d0p,(epsilon + B0p)**2)
	alpha1p = np.divide(d1p,(epsilon + B1p)**2)
	alpha2p = np.divide(d2p,(epsilon + B2p)**2)
	alphasump = alpha0p + alpha1p + alpha2p

	# ENO stencils weigths
	w0p = np.divide(alpha0p,alphasump)
	w1p = np.divide(alpha1p,alphasump)
	w2p = np.divide(alpha2p,alphasump)

	# Numerical Flux at cell boundary, $u_{i-1/2}^{+}$;
	hp = np.multiply(w0p,p0p) + np.multiply(w1p,p1p) + np.multiply(w2p,p2p)

	# Compute finite volume residual term, df/dx.
	res = (hp-np.roll(hp,1)+hn-np.roll(hn,1))/dx - S

	return res[0]

if __name__ =='__main__':
	run()