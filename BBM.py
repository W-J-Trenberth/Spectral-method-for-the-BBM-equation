# -*- coding: utf-8 -*-
"""
@author: William
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import animation

def main():
    
    N = 2**10
    dt = 0.005
    eps = 0.00001
    
    #The inital data
    #u_0 = 7/np.cosh(np.sqrt(1/(5.5*eps))*(x - 0.25))**2  #+ 6/np.cosh(np.sqrt(1/(6*eps))*(x - 0.35))**2
    u_0 = random_initial_data(N, 2.2)
    #u_0 = np.roll(np.loadtxt("sol_wave_amp_10.txt"), 500) + np.roll(np.loadtxt("sol_wave_amp_7.txt"), -250)
    
    #Animation hte solution
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 1), ylim=(-2,2))
    line, = ax.plot([], [], lw=2)
    animate = Animate(line,dt,u_0,eps)  
    anim = animation.FuncAnimation(fig, animate, frames=5000, interval=20)
    
    #Saving the animation
    anim.save("bbm_anim.mp4")



def BBM_solver(t, u_0, eps):
    '''A function to solve the BBM equation 
    $$\partial_t u + \partial_x u + u\partial_x u -\eps \partial_{xxt}u = 0$$.
    Using the Fourier analysis this equation can be written in the form 
    $$\partial_t u =F(u)$$. This function uses the fast Fourier transform to 
    compute $F(u)$ quickly solve_ivp to solve the resulting system of ODEs.
    
    Parameters
    -------------------------------------
    t: A time to evaluate the solution at.
    u_0: an array 
    eps: a number rpresenting the strength of the dispersion.
    
    Returns
    ------------------------------------
    out: the solution to BBM evaluated at time t starting from inital data u_0.
    '''
    
    #The nonlinearity of the ODE.
    def F(s,u):
        N=len(u)//2+ 1
        n = np.arange(0,N)
        phase = -np.complex(0,1)*2*np.pi*n/(1+eps*4*np.pi**2*n**2)
        fourier_Fu = phase*np.fft.rfft(u + u**2/2)
        Fu = np.fft.irfft(fourier_Fu)
        return Fu
    #Use solve_ivp to solve the ODE.
    out = np.reshape(solve_ivp(F, [0,t], u_0, t_eval = [t]).y, -1)
    
    return out

class Animate():
    '''Used to define an animate object used to animate the solution. These
    objects store the current u0 value, after BBM_solver is called, in a way 
    sort of replicating a static function variable. If you want to 
    compute the solution at time t_2 and have the value at t_1 you can go 
    from t_1 to t_2 instead of 0 to t_2 which would be inefficent.
    '''
    def __init__(self,line, dt, u_0, eps):
        self.line = line
        self.dt = dt
        self.u = u_0
        self.eps = eps
        self.x = np.linspace(0,1,len(u_0))
        
    def __call__(self,i):
        self.u = BBM_solver(self.dt, self.u, self.eps)
        self.line.set_data(self.x, self.u)
        return self.line
    
def random_initial_data(N,s):
    '''Using the DFT,  randomly generates a funcion of the form 
    $\sum\limits_{k=-N/2-1}^{N/2} \frac{g_k}{\langle k\rangle^s}e^{k\pi x}$
    where $g_k$ is a sequence of independent identically distributed 
    random variables and $\overline{g_k} = g_{-k} $.
    '''
    k = np.arange(1,N +1)
    Ff = (np.random.randn(N) + np.complex(0,1)*np.random.randn(N))/((k**2 + 1)**(s/2))
    f = np.fft.irfft(Ff)
    
    return 1000*f


if __name__ == "__main__": main()


    
    

