import numpy as np 
from numpy.fft import fft, fftfreq ,fftshift
from scipy.interpolate import interp1d

#------------ Spectral analysis---------------------------------

def fast_f_t(x : np.ndarray,y:np.ndarray, M:int = 500):
		t_interp = np.linspace(0, x[-1], M)  
		dt = t_interp[1] - t_interp[0]
		y = np.interp(t_interp,x, y )
		y -= np.mean(y)
		k = np.fft.fftfreq(M, d=dt)
		yw = np.fft.fft(y)
		return 2*np.pi*fftshift(k), fftshift(yw)*dt


def average_fft(x, y, Ms):
	spectra = []
	freqs_list = []

	for M in Ms:
		omega, A = fast_f_t(x, y, M)
		freqs_list.append(omega)
		spectra.append(A)

	omega_min = max(freqs[0] for freqs in freqs_list)     # límite inferior común
	omega_max = min(freqs[-1] for freqs in freqs_list)    # límite superior común
	N_common = max(len(f) for f in freqs_list)            # densidad similar a la mayor
	omega_common = np.linspace(omega_min, omega_max, N_common)


	spectra_interp = []
	for omega, A in zip(freqs_list, spectra):
		f_interp = interp1d(omega, A, kind='linear', bounds_error=False, fill_value=0.0)
		spectra_interp.append(f_interp(omega_common))

	A_avg = np.mean(spectra_interp, axis=0)

	return omega_common, A_avg

def correlation(a_0,t,initial):
	''' In a system with discrete spectrum, the spectral response id the fft of the correlations '''
	
	tau_values = np.linspace(0,t[-1],len(t)) #equal spacing, useful for fast fourier transform 
	state = np.asarray(initial)

	a_dag_0 = np.conjugate(np.transpose(a_0.reshape(-1,2,2),axes=(0,2,1))).reshape(-1,4) # need to transpose and conjugate first 
	
	interp_a_dag = interp1d(t,a_dag_0,axis=0, fill_value="extrapolate")
	a_dag =interp_a_dag(tau_values).reshape(-1,2,2)
	
	interp_a = interp1d(t,a_0,axis=0, fill_value="extrapolate")
	a_ttau =interp_a(t[-1]-tau_values).reshape(-1,2,2)

	return tau_values,np.einsum('i,tik,tkj,j->t',np.conjugate(state),a_dag,a_ttau,state)

# --------------------------- don't know how to call this , things That I will reuse and that makes sense to separate

from joblib import Parallel, delayed 

def paralelizar(parameter_list,f,ncores: int = 80):
	resultados = Parallel(n_jobs=ncores, backend='loky')(
		delayed(f)(param) for param in parameter_list
	)
	return resultados


from logistics_DDE import new_cav_model

def run_simulation(gamma,tau,phi,Omega,t_max,dt,initial,Ms = np.arange(5000,5200,1)):
	cav = new_cav_model(gamma,phi,tau,Omega)
	cav.evolve(t_max,dt)
	t,e = cav.excited_state(np.asarray(initial))
	tau_p,corr = correlation(cav.a_out_array,t,initial)
	return t,e, cav.a_out_array,cav.s_array

def observable(m,initial):
	state=np.asarray(initial)
	if m.shape[-1]==4:
		m_2 = m.reshape(-1,2,2)
		m_dag = np.conjugate(np.transpose(m_2,axes=(0,2,1)))
		return np.einsum('i,tik,tkj,j->t',np.conjugate(state),m_dag,m_2,state)
	else:
		m_dag = np.conjugate(np.transpose(m,axes=(0,2,1)))
		return np.einsum('i,tik,tkj,j->t',np.conjugate(state),m_dag,m,state)
	
def fft_matrix(t,a,Ms=np.arange(5000,5200,1)):
    ''' enters as a (n,4), leaves as a (n,2,2)'''
    a_w = []
    w,a0 = average_fft(t,a[:,0],Ms)
    a_w.append(a0)
    for n in range(1,4):
        _,a_dummie = average_fft(t,a[:,n],Ms)
        a_w.append(a_dummie)
    return w, (np.asarray(a_w).T).reshape(-1,2,2)

