import numpy as np
from scipy.special import expit

def g_function(w, D, Label, n):
    N1, P = D.shape
    N1 = N1 + 1
    Xh = np.r_[D, np.ones((1, P))]
    W = w.reshape(N1, n, order='F')
    g = np.zeros((N1*n, 1))
    for i in range(n):
        wi = W[:, i]
        gi = np.zeros((N1,))
        for p in range(P):
            y = Label[p][i]
            xp = Xh[:, p]
            tp = expit(xp.dot(wi))
            t0 = sum(expit(xp.dot(W)))
            gi = gi + 2*(tp/t0 - y)*(tp/t0 - (tp/t0)**2)*xp
        gi = gi/P
        g[i*N1:(i+1)*N1,0] = gi
    return g

