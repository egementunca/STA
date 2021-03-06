import numpy as np
from numpy import einsum
from scipy.linalg import expm
from numpy.linalg import eig, norm
import numba as nb

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

trace = lambda a: np.einsum('ii', a)
kron = lambda a, b: np.einsum('ij,kl -> ikjl', a, b).reshape(a.shape[0]*b.shape[0],a.shape[1]*b.shape[1])
matmul = lambda a, b: np.einsum('ij,jk', a, b)
elementwise = lambda a, b : np.einsum('i,i->i', a, b)

#CONSTANTS
hbar = 1
Ns = 3
Ds = 8
T = 10
kb = 1
beta = 1/(kb*T)
Dp1 = 4
Dp2 = 6
Dp3 = 1000

#Time Params
dt = 2e-3
Tau = 1
T_0 = 0
t,s,St,Lt = TimeVecV4(T_0, dt, Tau)

#Pauli Matrices
sx = 0.5 * np.array([[0,1],[1,0]])
sy = 0.5 * -1j *np.array([[0,1],[-1,0]])
sz = 0.5 * np.array([[1,0],[0,-1]])
I = np.eye(2)

sx3 = kron(kron(sx,I),I) + kron(kron(I,I),sx) + kron(kron(I,sx),I)

opz = np.zeros((Dp1,Dp2*Dp3,Lt))
opy = np.zeros((Dp1,Dp2*Dp3,Lt),dtype=np.complex128)
opx = np.zeros((Dp1,Dp2*Dp3,Lt))

z = np.zeros((Dp1,Dp2*Dp3,Lt))
y = np.zeros((Dp1,Dp2*Dp3,Lt),dtype=np.complex128)
x = np.zeros((Dp1,Dp2*Dp3,Lt))

zij = np.zeros((Dp1,Dp2*Dp3,Lt))
yij = np.zeros((Dp1,Dp2*Dp3,Lt),dtype=np.complex128)
xij = np.zeros((Dp1,Dp2*Dp3,Lt))



f1 = lambda rho, op: (1/3) * ( (trace(matmul(rho,kron(kron(op,I),I)))**2) + (trace(matmul(rho,kron(kron(I,I),op)))**2) + (trace(matmul(rho,kron(kron(I,op),I)))**2))
f2 = lambda rho, op: (1/3) * ( (trace(matmul(rho,kron(kron(op,I),I)))) + (trace(matmul(rho,kron(kron(I,I),op)))) + (trace(matmul(rho,kron(kron(I,op),I)))))
f3 = lambda rho, op: (1/3) * ( (trace(matmul(rho,kron(kron(op,op),I)))) + (trace(matmul(rho,kron(kron(op,I),op)))) + (trace(matmul(rho,kron(kron(I,op),op)))))

f1_eig = lambda psi, op: (1/3) * ( matmul(psi.conj().T, matmul(kron(kron(op,I),I), psi))[0][0]**2 + matmul(psi.conj().T, matmul(kron(kron(I,I),op), psi))[0][0]**2 + matmul(psi.conj().T, matmul(kron(kron(I,op),I), psi))[0][0]**2 )
f2_eig = lambda psi, op: (1/3) * ( matmul(psi.conj().T, matmul(kron(kron(op,I),I), psi))[0][0] + matmul(psi.conj().T, matmul(kron(kron(I,I),op), psi))[0][0] + matmul(psi.conj().T, matmul(kron(kron(I,op),I), psi))[0][0] )
f3_eig = lambda psi, op: (1/3) * ( matmul(psi.conj().T, matmul(kron(kron(op,op),I), psi))[0][0] + matmul(psi.conj().T, matmul(kron(kron(op,I),op), psi))[0][0] + matmul(psi.conj().T, matmul(kron(kron(I,op),op), psi))[0][0] )

def realizationCreator(q,j_pool):
	for i in range(Dp2*Dp3):
		for t in range(Lt):
			q[:,:,i,t] += j_pool[i//6].conj().T
	return q

h0, h2 = 0, 1
H1 = np.array([0.1, 0.5, 1.0, 2.0])
h = [PolyDrivingV4(Tau, t, h0, h2)[0]]
	
mj = -2* np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1]])
	
#can be problematic since gaussian dist is not bounded
j_pool = np.random.normal(loc=0, scale=1/3, size=(Dp3,1,3))
#j_pool = np.array([[1,1,1],[1,1,-1]]).reshape(Dp3,1,3)
	
r = np.array([mj * j for j in j_pool]).reshape(Dp2*Dp3,1,3)
q = np.array([r]).conj().T @ h
realizations = realizationCreator(q,j_pool)

def sup_pos(v1, v2):
	v = v1 + v2
	n = norm(v)
	return (v/n).reshape(len(v1),1)

nb.njit(parallel=True)
def main():

	jh_ratio = np.zeros(Dp1*Dp2*Dp3*Lt).reshape(Dp1,Dp2*Dp3,Lt)

	for p1,h1 in enumerate(H1):
		for p2 in range(0,Dp2*Dp3):
			for i in range(0,Lt):

				j = realizations[:,0,p2,i]
				jh_ratio[p1,p2,i] = j.mean()/h1

				H0f= j[0]*kron(kron(sz,sz),I)+j[1]*kron(I,kron(sz,sz))+j[2]*kron(kron(sz,I),sz)+ h1*sx3
				#rho = GibbsV4(beta, H0f) #gibbs solution
				w, v = eig(H0f) #first excited state solution
				idx = np.argsort(w)
				psi = sup_pos(v[:,idx[0]], v[:,idx[1]])


				opz[p1,p2,i] = f1_eig(psi,sz)
				opy[p1,p2,i] = f1_eig(psi,sy)
				opx[p1,p2,i] = f1_eig(psi,sx)


				z[p1,p2,i] = f2_eig(psi,sz)
				y[p1,p2,i] = f2_eig(psi,sy)
				x[p1,p2,i] = f2_eig(psi,sx)


				zij[p1,p2,i] = f3_eig(psi,sz)
				yij[p1,p2,i] = f3_eig(psi,sy)
				xij[p1,p2,i] = f3_eig(psi,sx)
		

	np.savez('big_data.npz', realizations=realizations, jh=jh_ratio, opx=opx,opy=opy,opz=opz, x=x, y=y, z=z, xij=xij, yij=yij, zij=zij)

	return True

def analyze_speed():

	import cProfile
	import pstats

	with cProfile.Profile() as pr:
		main()

	stats = pstats.Stats(pr)
	stats.sort_stats(pstats.SortKey.TIME)
	stats.dump_stats(filename='test0.prof')

	return True