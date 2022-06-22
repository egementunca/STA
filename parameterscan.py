import numpy as np
from numpy import kron, trace
from scipy.linalg import expm

def GibbsV4(beta, H):
	rho = expm(-beta*H) / trace(expm(-beta*H))
	return rho

def TimeVecV4(T_0, dt, Tau):

	t = np.arange(T_0,Tau,dt)
	s = t/Tau
	St = np.shape(t)
	Lt = len(t)

	return t, s, St, Lt

def PolyDrivingV4(Tau, t, wi, wf):

	wfi = wf - wi
	w = wi + (10*wfi*((t/Tau)**3)) - (15*wfi*((t/Tau)**4)) + (6*wfi*((t/Tau)**5))
	dw = ( (30*wfi*((t/Tau)**2)) - (60*wfi*((t/Tau)**3)) + (30*wfi*((t/Tau)**4)) )/Tau
	return w, dw


#CONSTANTS
hbar = 1
Ns = 3
Ds = 8
T = 10
kb = 1
beta = 1/(kb*T)
Dp1 = 10
Dp2 = 6
Dp3 = 100

#Time Params
dt = 1e-2
Tau = 1
T_0 = 0
t,s,St,Lt = TimeVecV4(T_0, dt, Tau)

#Pauli Matrices
sx = 0.5 * np.array([[0,1],[1,0]])
sy = 0.5 * -1j *np.array([[0,1],[-1,0]])
sz = 0.5 * np.array([[1,0],[0,-1]])
I = np.eye(2)

sx3 = kron(kron(sx,I),I) + kron(kron(I,I),sx) + kron(kron(I,sx),I)

H0f = np.zeros((Ds,Ds,Lt))
opz = np.zeros((Dp1,Dp2,Dp3,Lt))
opy = np.zeros((Dp1,Dp2,Dp3,Lt),dtype=np.complex128)
opx = np.zeros((Dp1,Dp2,Dp3,Lt))

z = np.zeros((Dp1,Dp2,Dp3,Lt))
y = np.zeros((Dp1,Dp2,Dp3,Lt),dtype=np.complex128)
x = np.zeros((Dp1,Dp2,Dp3,Lt))

zij = np.zeros((Dp1,Dp2,Dp3,Lt))
yij = np.zeros((Dp1,Dp2,Dp3,Lt),dtype=np.complex128)
xij = np.zeros((Dp1,Dp2,Dp3,Lt))


f1 = lambda rho, op: (1/3) * ( (trace(rho @ kron(kron(op,I),I))**2) + (trace(rho @ kron(kron(I,I),op))**2) + (trace(rho @ kron(kron(I,op),I))**2))
f2 = lambda rho, op: (1/3) * ( (trace(rho @ kron(kron(op,I),I))) + (trace(rho @ kron(kron(I,I),op))) + (trace(rho @ kron(kron(I,op),I))))
f3 = lambda rho, op: (1/3) * ( (trace(rho @ kron(kron(op,op),I))) + (trace(rho @ kron(kron(op,I),op))) + (trace(rho @ kron(kron(I,op),op))))

def main():
	for p1 in range(0,Dp1):
		for p2 in range(0,Dp2):
			for p3 in range(0,Dp3):
				
				j = np.random.uniform(low=-1.0, high=1.0, size=(1,3))
				h0, h1, h2 = 0, 0.1*p1, 1
				h = [PolyDrivingV4(Tau, t, h0, h2)[0]]
	
				mj = -1* np.array([
					[1,0,0],
					[0,1,0],
					[0,0,1],
					[1,1,0],
					[0,1,1],
					[1,0,1]])
	
				jj = j.conj().T + ((mj[p2,:] * j).conj().T @ h)
	
				for i in range(0,Lt):
					
					H0f= jj[0,i]*kron(kron(sz,sz),I)+jj[1,i]*kron(I,kron(sz,sz))+jj[2,i]*kron(kron(sz,I),sz)+ h1*sx3
					rho = GibbsV4(beta, H0f)
					
					opz[p1,p2,p3,i] = f1(rho,sz)
					opy[p1,p2,p3,i] = f1(rho,sy)
					opx[p1,p2,p3,i] = f1(rho,sx)
	
					
					z[p1,p2,p3,i]= f2(rho,sz)
					y[p1,p2,p3,i]=f2(rho,sy)
					x[p1,p2,p3,i]=f2(rho,sx)
	
					
					zij[p1,p2,p3,i]=f3(rho,sz)
					yij[p1,p2,p3,i]=f3(rho,sy)
					xij[p1,p2,p3,i]=f3(rho,sx)

	np.savez('data.npz', opx=opx,opy=opy,opz=opz, x=x, y=y, z=z, xij=xij, yij=yij, zij=zij)

	return True

def analyze_speed():

	import cProfile
	import pstats

	with cProfile.Profile() as pr:
		main()

	stats = pstats.Stats(pr)
	stats.sort_stats(pstats.SortKey.TIME)
	stats.dump_stats(filename='test.prof')

	return True

analyze_speed()
