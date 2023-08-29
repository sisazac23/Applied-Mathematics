import numpy as np

def modified_euler(A,y0,t,h):
    y = np.zeros((len(t),len(y0)))
    y[0] = y0
    for i in range(1,len(t)):
        y[i] = y[i-1] + h/2*(A@y[i-1] + A@(y[i-1] + h*A@y[i-1]))
    return y