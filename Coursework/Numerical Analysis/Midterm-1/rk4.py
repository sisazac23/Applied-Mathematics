import numpy as np

def rk4(A,y0,t,h):
    y = np.zeros((len(t),len(y0)))
    y[0] = y0
    for i in range(len(t)-1):
        k1 = h*A@y[i]
        k2 = h*A@(y[i]+k1/2)
        k3 = h*A@(y[i]+k2/2)
        k4 = h*A@(y[i]+k3)
        y[i+1] = y[i] + (k1+2*k2+2*k3+k4)/6
    return y