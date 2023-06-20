import numpy as np
from scipy import special as spc

def getTransMat(k, k_max='auto', TensorCorePad=True):
    
    # Defining the radius
    # Be careful that k should be an odd number
    R = k // 2
    
    # Computing k_max if needed using the Nyquist frequency
    if k_max == 'auto':
        k_max = np.pi * (k / 2.)
        
    # Get all the k_mj such that K_mj < k_max
    k_mj = []
    m = 0
    while True:
        
        # Scipy ignores 0's at x=0 
        if m == 0:
            k_m = [0]
        else:
            k_m = []
            
        j = 0
        zeros = spc.jnp_zeros(m, 10)
        while True:
            
            # Check if we need more 0's
            # If yes, add the 5 next 0's
            if j >= zeros.shape[0]:
                zeros = spc.jnp_zeros(m, j+5)
                
            # If k_mj > k_max, stop
            if zeros[j] > k_max:
                break
            # Otherwise, add k_mj to the list
            else:
                k_m.append(zeros[j])
                j += 1
                
        # If there is no 0's s.t. k_mj <= k_max for this m, we can stop
        if j == 0:
            break
        # Otherwise, continue with the next m
        else:  
            k_mj.append(k_m)
            m += 1
    
    w = len(k_mj) ; h = max([len(x) for x in k_mj])
    if not TensorCorePad:
        K = np.zeros(shape=(w, h), dtype=np.float32)
    else:
        while True:
            if (w % 8) == 0:
                break
            else:
                w += 1
        while True:
            if (h % 8) == 0:
                break
            else:
                h += 1
        K = np.zeros(shape=(w, h), dtype=np.float32)
    
    # MASK keeps track of which values should be used or not
    # (i.e., if those are the result of padding or not)
    MASK = np.full(shape=(w, h), fill_value=False)
    for i, x in enumerate(k_mj):
        K[i,:len(x)] = x
        MASK[i,:len(x)] = True
        
    # Building the grid
    _x = np.linspace(-1, 1, k)
    grid = np.meshgrid(_x, _x)
    theta = np.angle(grid[0][:,:] + 1j * grid[1][:,:])
    
    # Compute the normalization factors
    J = np.zeros(shape=(R+1, w*h), dtype=np.float32)
    P = np.linspace(0, R, R+1, dtype=np.float32)
    
    for m in range(w):
        for j in range(h):
            if MASK[m,j]:
                J[:, j+m*h] = spc.jv(m, K[m,j] * P[:])
    
    A = np.sqrt(2. * np.pi * np.matmul(P, np.square(J)))
    A = np.reshape(A, (w, h))
    
    T = np.zeros(shape=(k, k, w, h), dtype=np.complex64)
    for m in range(w):
        for j in range(h):
            if not MASK[m,j]:
                break
            for _j in range(k):
                for _i in range(k):
                    if np.sqrt(grid[0][_i, _j]**2 + grid[1][_i, _j]**2) <= 1.:
                        T[_i, _j, m, j] = spc.jv(m, K[m,j] * np.sqrt(grid[0][_i,_j]**2 + grid[1][_i,_j]**2))
                        T[_i, _j, m, j] *= np.exp(-1j * m * theta[_i,_j]) * A[m,j]**-1
                        
    return T, MASK