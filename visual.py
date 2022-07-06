import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

data = np.load('data_eig_method.npz')#mixed realizations dp3:100, first excited state
#data = np.load('test_data_eig_method.npz') # given realizations dp3:2, first excited state
#data = np.load('test_data.npz') #given realization dp3:2, gibbs
#data = np.load('data.npz')# mixed realizations dp3:100, gibbs

realizations = data['realizations']
jh = data['jh']
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
Dp2 = 6 #NUMBER OF POSSIBLE FLIPS
Dp3 = 100 #SIZE OF J POOL
		#DP2 * DP3 : NUMBER OF REALIZATIONS 

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

def average_general(a):
	size = Lt
	x = np.zeros(size)
	for i in range(size):
		x[i] = np.sum(a[:,:,i])/(Dp1*Dp2*Dp3)
	return x

#A General Average
Aopz=average_general(opz)
Aopy=average_general(opy)
Aopx=average_general(opx)

Az=average_general(z)
Ay=average_general(y)
Ax=average_general(x)

Azij=average_general(zij)
Ayij=average_general(yij)
Axij=average_general(xij)

#average for realizations with same h value
def average(a):

	size = Lt
	x = np.zeros(size)
	for i in range(size):
		x[i] = np.sum(a[:,i])/(Dp2*Dp3)
	return x

def show_realization(real_no):

	op = (opx[:,real_no], opy[:,real_no], opz[:,real_no])
	op_ij = (xij[:,real_no], yij[:,real_no], zij[:,real_no])
	spins = (x[:,real_no], y[:,real_no], z[:,real_no])
	labels = ('x', 'y', 'z')

	observable = (op, spins, op_ij)
	obs_name = ('q', r'<\sigma>', r'<\sigma^i\sigma^j>')

	fig, axs = plt.subplots(figsize=(19.20, 10.80), dpi=100 ,nrows=3, ncols=3, sharex=True)
	for obs in range(len(observable)):
		for indice in range(3):
			for h_step in range(10):
				axs[indice, obs].plot(s,observable[obs][indice][h_step],label='h: {}'.format(np.around(0.1*(h_step+1),decimals=2)))
				axs[indice, obs].set_title(r'${obs_name}_{label}$'.format(obs_name=obs_name[obs],label=labels[indice]))
				axs[indice, obs].set_xlabel(r'$t/\tau$')
	plt.suptitle('j from: {} to {}'.format(realizations[:,0,real_no,0],realizations[:,0,real_no,-1]))
	axs[1,2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
	#plt.savefig('h_compared_fig_eig_{}.png'.format(real_no))
	plt.show()

def show():

	op = (opx, opy, opz)
	op_ij = (xij, yij, zij)
	spins = (x, y, z)
	observable = (op, spins, op_ij)

	labels = ('x', 'y', 'z')
	obs_name = ('q', r'<\sigma>', r'<\sigma^i\sigma^j>')

	fig, axs = plt.subplots(figsize=(19.20, 10.80), dpi=100 ,nrows=3, ncols=3, sharex=True)
	for obs in range(len(observable)):
		for indice in range(3): # indices of observables x,y,z
			for h_step in range(10):
				axs[indice, obs].plot(s, average(observable[obs][indice][h_step]), label='h: {}'.format(np.around(0.1*(h_step+1),decimals=2)))
				axs[indice, obs].set_title(r'${obs_name}_{label}$'.format(obs_name=obs_name[obs], label=labels[indice]))
				axs[indice, obs].set_xlabel(r'$t/\tau$')
	axs[1,2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.suptitle('Average of {} Realizations'.format(Dp2*Dp3))
	plt.show()

def old_show():
	##FIGURES
	fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
	
	axs[0].plot(s, Aopx)
	axs[0].set_ylabel(r'$q_{\sigma_x}$')
	
	axs[1].plot(s, Aopy)
	axs[1].set_ylabel(r'$q_{\sigma_y}$')
	
	axs[2].plot(s, Aopz)
	axs[2].set_ylabel(r'$q_{\sigma_z}$')
	axs[2].set_xlabel(r'$t/\tau}$')
	
	fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
	
	axs[0].plot(s, Ax)
	axs[0].set_ylabel(r'$<\sigma_x>$')
	
	axs[1].plot(s, Ay)
	axs[1].set_ylabel(r'$<\sigma_y>$')
	
	axs[2].plot(s, Az)
	axs[2].set_ylabel(r'$<\sigma_z>$')
	axs[2].set_xlabel(r'$t/\tau}$')
	
	fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
	
	axs[0].plot(s, Axij)
	axs[0].set_ylabel(r'$<\sigma^i_x\sigma^j_x>$')
	
	axs[1].plot(s, Ayij)
	axs[1].set_ylabel(r'$<\sigma^i_y\sigma^j_y>$')
	
	axs[2].plot(s, Azij)
	axs[2].set_ylabel(r'$<\sigma^i_z\sigma^j_z>$')
	axs[2].set_xlabel(r'$t/\tau}$')
	
	#plt.show()
	
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


'''
show realization old code

	fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
	for p in range(3):
		for i in range(10):
			axs[p].plot(s,op[p][i],label='j from: {} to {} with h: {}'.format(realizations[:,0,real_no,0],realizations[:,0,real_no,-1],0.1*(i+1)))
			axs[p].set_ylabel(r'$q_{label}$'.format(label=labels[p]))
			plt.legend()

	fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
	for p in range(3):
		for i in range(10):
			axs[p].plot(s,spins[p][i],label='j from: {} to {} with h: {}'.format(realizations[:,0,real_no,0],realizations[:,0,real_no,-1],0.1*(i+1)))
			axs[p].set_ylabel(r'$<\sigma_{label}>$'.format(label=labels[p]))
			plt.legend()
	
	fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True)
	for p in range(3):
		for i in range(10):
			axs[p].plot(s,op_ij[p][i],label='j from: {} to {} with h: {}'.format(realizations[:,0,real_no,0],realizations[:,0,real_no,-1],0.1*(i+1)))
			axs[p].set_ylabel(r'$<\sigma^i_{label}\sigma^j_{label}>$'.format(label=labels[p]))
			plt.legend()
	
	'''
