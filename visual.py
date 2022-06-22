import numpy as np
import matplotlib.pyplot as plt

#opz = np.load('opz.npy')
#opy = np.load('opy.npy')
#opx = np.load('opx.npy')
#z   = np.load('z.npy')
#x   = np.load('x.npy')
#y   = np.load('y.npy')
#zij = np.load('zij.npy')
#xij = np.load('xij.npy')
#yij = np.load('yij.npy')

data = np.load('data.npz')

opz = data['opz']
opy = data['opy']
opx = data['opx']
z   = data['z']
x   = data['x']
y   = data['y']
zij = data['zij']
xij = data['xij']
yij = data['yij']

Dp1 = 10
Dp2 = 6
Dp3 = 100


def TimeVecV4(T_0, dt, Tau):

	t = np.arange(T_0,Tau,dt)
	s = t/Tau
	St = np.shape(t)
	Lt = len(t)

	return t, s, St, Lt


#Time Params
dt = 1e-2
Tau = 1
T_0 = 0
t,s,St,Lt = TimeVecV4(T_0, dt, Tau)



def average(a):
	size = np.shape(opz)[-1]
	x = np.zeros(size)
	for i in range(size):
		x[i] = np.sum(a[:,:,:,i])/(Dp1*Dp2*Dp3)
	return x


Aopz=average(opz)
Aopy=average(opy)
Aopx=average(opx)

Az=average(z)
Ay=average(y)
Ax=average(x)

Azij=average(zij)
Ayij=average(yij)
Axij=average(xij)

##FIGURES
fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(16,9), dpi=120)

axs[0].plot(s, Aopx)
axs[0].set_ylabel(r'$q_{\sigma_x}$')

axs[1].plot(s, Aopy)
axs[1].set_ylabel(r'$q_{\sigma_y}$')

axs[2].plot(s, Aopz)
axs[2].set_ylabel(r'$q_{\sigma_z}$')
axs[2].set_xlabel(r'$t\tau}$')

fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(16,9), dpi=120)

axs[0].plot(s, Ax)
axs[0].set_ylabel(r'$<\sigma_x>$')

axs[1].plot(s, Ay)
axs[1].set_ylabel(r'$<\sigma_y>$')

axs[2].plot(s, Az)
axs[2].set_ylabel(r'$<\sigma_z>$')
axs[2].set_xlabel(r'$t\tau}$')

fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(16,9), dpi=120)

axs[0].plot(s, Axij)
axs[0].set_ylabel(r'$<\sigma^i_x\sigma^j_x>$')

axs[1].plot(s, Ayij)
axs[1].set_ylabel(r'$<\sigma^i_y\sigma^j_y>$')

axs[2].plot(s, Azij)
axs[2].set_ylabel(r'$<\sigma^i_z\sigma^j_z>$')
axs[2].set_xlabel(r'$t\tau}$')

plt.show()

fig, axs = plt.subplots(nrows=3, ncols=1)
axs[0].hist(opx.flatten(),bins=100)
axs[0].set_xlabel(r'$q_{\sigma_x}$')
axs[1].hist(np.real(opy).flatten(),bins=100)
axs[1].set_xlabel(r'$q_{\sigma_y}$')
axs[2].hist(opz.flatten(),bins=100)
axs[2].set_xlabel(r'$q_{\sigma_z}$')

fig, axs = plt.subplots(nrows=3, ncols=1)
axs[0].hist(x.flatten(),bins=100)
axs[0].set_xlabel(r'$<\sigma_x>$')
axs[1].hist(np.real(y).flatten(),bins=100)
axs[1].set_xlabel(r'$<\sigma_y>$')
axs[2].hist(z.flatten(),bins=100)
axs[2].set_xlabel(r'$<\sigma_z>$')

fig, axs = plt.subplots(nrows=3, ncols=1)
axs[0].hist(xij.flatten(),bins=100)
axs[0].set_xlabel(r'$<\sigma^i_x\sigma^j_x>$')
axs[1].hist(np.real(yij).flatten(),bins=100)
axs[1].set_xlabel(r'$<\sigma^i_y\sigma^j_y>$')
axs[2].hist(zij.flatten(),bins=100)
axs[2].set_xlabel(r'$<\sigma^i_z\sigma^j_z>$')

plt.show()


	