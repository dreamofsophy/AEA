import numpy as np
from math import *
from scipy import io
import matplotlib.pyplot as plt
from scipy import signal

def hog(Im, d, B):
    t = floor(d/2)
    (N,M) = Im.shape
    k1 = (M - d)/t
    c1 = ceil(k1)
    k2 = (N -d)/t
    c2 = ceil(k2)
    if c1 - k1 > 0:
        M1 = d + t*c1
        Im = np.c_[Im, np.fliplr(Im[:,(2*M - M1):])]
    if c2 - k2 > 0:
        N1 = d + t*c2
        Im = np.r_[Im, np.flipud(Im[(2*N - N1):, :])]
    (N,M) = Im.shape
    nx1 = np.arange(0, M-d+1, t)
    nx2 = np.arange(d-1, M, t)
    ny1 = np.arange(0, N-d+1, t)
    ny2 = np.arange(d-1, N, t)
    Lx = len(nx1)
    Ly = len(ny1)
    hz = Lx*Ly*B
    H = np.zeros((hz,1))
    Im = Im.astype(np.float32)
    k = 1
    hx = np.array([-1, 0, 1]).reshape(1,3)
    hy = -hx.T
    grad_xr = signal.convolve(Im, hx, mode='same')
    grad_yu = signal.convolve(Im, hy, mode='same')    
    magnit = np.sqrt(np.square(grad_xr) + np.square(grad_yu)) 
    angles = np.arctan2(grad_yu, grad_xr)
    for m in range(Lx):
        for n in range(Ly):
            angles2 = angles[ny1[n]:ny2[n]+1, nx1[m]:nx2[m]+1]
            magnit2 = magnit[ny1[n]:ny2[n]+1, nx1[m]:nx2[m]+1]
            v_angles = angles2.flatten('F') # the order is not
            v_magnit = magnit2.flatten('F')
            K = len(v_angles)
            Bin = 0
            h2 = np.zeros((B,1))
            delt = 2*np.pi/B
            for ang_lim in np.arange(-np.pi+ delt, np.pi+delt,delt ):
                for i in range(K):
                    if v_angles[i] < ang_lim:
                        v_angles[i] = 100
                        h2[Bin, :] = h2[Bin, :] + v_magnit[i]
                Bin = Bin + 1
            h2 = h2/(np.linalg.norm(h2) + 0.01)
            H[(k-1)*B: k*B,:] = h2
            k = k+1
    return H
