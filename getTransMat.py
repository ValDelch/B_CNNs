import numpy as np
from scipy import special

def getTransMat(k, k_max='auto', TensorCorePad=True):

    R = k // 2

    if k_max == 'auto':
        k_max = np.pi * k/2.
    
    # Get all the k_mj such that k_mj < k_max
    k_mj = []
    m = 0
    while True:
        if m != 1:
            _k_mj = [0.]
        else:
            _k_mj = []
        j = 0
        zeros = special.jnp_zeros(m, 10)
        while True:
            if j >= zeros.shape[0]:
                zeros = special.jnp_zeros(m, int(1.5*j))
            if zeros[j] >= k_max:
                break
            _k_mj.append(zeros[j])
            j += 1
        if _k_mj == [0.]:
            break
        k_mj.append(_k_mj)
        m += 1

    Nyquist_m_max = len(k_mj) - 1
    if not TensorCorePad:
        K = np.zeros((Nyquist_m_max+1, max([len(x) for x in k_mj])), dtype=np.double)
    else:
        ini_w = Nyquist_m_max+1
        ini_h = max([len(x) for x in k_mj])
        while True:
            if ini_w % 8 == 0:
                break
            else:
                ini_w += 1
        while True:
            if ini_h % 8 == 0:
                break
            else:
                ini_h += 1
        K = np.zeros((ini_w, ini_h), dtype=np.double)

    for i, x in enumerate(k_mj):
        K[i,:len(x)] = x

    m_max, j_max = np.shape(K)
    m_max -= 1 ; j_max -= 1

    # Build the grid
    _x = np.linspace(-1, 1, k)
    grid = np.meshgrid(_x, _x)
    theta = np.angle(grid[0][:,:] + 1j * grid[1][:,:])

    # Compute the normalization factors A_mj
    J = np.zeros((R+1, (j_max+1)*(m_max+1)))
    P = np.linspace(0, R, R+1)

    for m in range(0, m_max+1):
        for j in range(0, j_max+1):
            if K[m,j] == 0. and j > 0:
                break
            J[:,j+m*(j_max+1)] = special.jv(m, K[m,j] * P[:])

    A = np.sqrt(2. * np.pi * np.matmul(P, np.square(J)))
    A = np.reshape(A, (m_max+1, j_max+1))

    # Set A_mj = 1 when it equals 0, to avoid zero divisions
    for m in range(2, m_max+1):
        A[m,0] = 1.

    n_coeffs = np.count_nonzero(A)

    # Compute transMat
    transMat = np.zeros((k, k, m_max+1, j_max+1), dtype=np.complex64)
    for m in range(0, m_max+1):
        for j in range(0, j_max+1):
            if A[m,j] == 0.:
                break
            for _j in range(k):
                for _i in range(k):
                    if np.sqrt(grid[0][_i,_j]**2 + grid[1][_i,_j]**2) > 1:
                        continue
                    transMat[_i, _j, m, j] = special.jv(m, K[m,j] * np.sqrt(grid[0][_i,_j]**2 + grid[1][_i,_j]**2))
                    transMat[_i, _j, m, j] *= np.exp(-1j * m * theta[_i,_j]) * A[m,j]**-1

    return transMat, m_max, j_max, Nyquist_m_max, K
