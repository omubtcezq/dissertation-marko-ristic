import numpy as np
from phe import paillier

import encryption as enc
import plotting as pltng

SAVE_FIGS = True
pltng.init_matplotlib_params(SAVE_FIGS, True)

"""
 
  ######  #### ##     ##    ##     ## ######## ##       ########  
 ##    ##  ##  ###   ###    ##     ## ##       ##       ##     ## 
 ##        ##  #### ####    ##     ## ##       ##       ##     ## 
  ######   ##  ## ### ##    ######### ######   ##       ########  
       ##  ##  ##     ##    ##     ## ##       ##       ##        
 ##    ##  ##  ##     ##    ##     ## ##       ##       ##        
  ######  #### ##     ##    ##     ## ######## ######## ##        
 
"""
class SimSensor:
    def __init__(self, F, Q, H, R, x, P):
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R
        self.x = x
        self.P = P
        return
    
    def estimate(self, gt):
        z = gt + np.random.multivariate_normal(np.zeros(2), self.R)
        self.predict()
        self.update(z)
        #Pinv = np.linalg.inv(self.P)
        #Pinvx = Pinv@self.x
        return self.x, self.P
    
    def predict(self):
        self.x = self.F@self.x
        self.P = self.Q + (self.F@self.P@self.F.T)
        return
    
    def update(self, z):
        S = (self.H@self.P@self.H.T) + self.R
        invS = np.linalg.inv(S)
        K = self.P@self.H.T@invS
        self.x = self.x + K@(z - self.H@self.x)
        self.P = self.P - (K@S@K.T)
        return

# SecFCI 2 sensor list intersection approx (l2 should already be reversed)
def intersect_approx_bsearch(l1, l2, discStep):
    # l1 < l2 to start with always
    startDiff = True
    interceptInd, approx = listIntersectSup(l1, l2, 0, startDiff)
    if approx:
        return (interceptInd - 0.5)*discStep
    else:
        return interceptInd*discStep

def listIntersectSup(l1, l2, index, startDiff):
    n = len(l1)
    # Base case
    if n == 1:
        if l1[0] == l2[0]:
            return (index, False)
        elif (l1[0] < l2[0]) == startDiff:
            return (index + 1, True)
        elif (l1[0] < l2[0]) != startDiff:
            return (index, True)
    # Single recurse go left or right depending on whether the sign has changed
    mid = n//2
    if l1[mid] == l2[mid]:
        return (index + mid, False)
    elif (l1[mid] < l2[mid]) == startDiff:
        return listIntersectSup(l1[mid:], l2[mid:], index+mid, startDiff)
    elif (l1[mid] < l2[mid]) != startDiff:
        return listIntersectSup(l1[:mid], l2[:mid], index, startDiff)

def copy_mat_into_mat(in_mat, out_mat, in_dim_i, in_dim_j, out_start_i, out_start_j):
    for i in range(in_dim_i):
        for j in range(in_dim_j):
            out_mat[out_start_i+i][out_start_j+j] = in_mat[i][j]

def copy_vec_into_mat(in_vec, out_mat, in_dim, out_start_i, out_start_j, as_col_vec=True):
    if as_col_vec:
        for i in range(in_dim):
            out_mat[out_start_i+i][out_start_j] = in_vec[i]
    else:
        for j in range(in_dim):
            out_mat[out_start_i][out_start_j+j] = in_vec[j]

def copy_vec_into_vec(in_vec, out_vec, in_dim, out_start):
    for i in range(in_dim):
        out_vec[out_start+i] = in_vec[i]

def omega_partials(sensor_lists):
    n = len(sensor_lists)

    # Linear system dimensions
    res = np.zeros(n*(n-1))
    mat = np.zeros((n*(n-1),n+(n-1)*(n-1)))

    # Get computable point, trivial points and differences
    for i in range(n-1):
        omega = omega_2sen(sensor_traces[i], sensor_traces[i+1])
        point = np.array([0]*i + [omega, 1-omega] + [0]*(n-2-i))
        known_points = []
        for j in range(n):
            if j==i or j==i+1:
                continue
            p = np.zeros(n)
            p[j] = 1
            known_points.append(p)
        directions = [p-point for p in known_points]
        #print(point,'')
        #print(known_points,'')
        #print(directions,'')

        # Populate linear system matrices
        copy_vec_into_vec(-point, res, n, i*n)
        copy_mat_into_mat(-np.eye(n), mat, n, n, i*n, 0)
        for ind,d in enumerate(directions):
            copy_vec_into_mat(d, mat, n, i*n, n+(n-2)*i+ind)
    #print(res,'')
    #print(mat,'')

    # Solve system
    sol_vec, err, rnk, sing = np.linalg.lstsq(mat, res)
    #print(err,')
    return sol_vec[:n]

"""
 
  ######  #### ##     ##  ######  
 ##    ##  ##  ###   ### ##    ## 
 ##        ##  #### #### ##       
  ######   ##  ## ### ##  ######  
       ##  ##  ##     ##       ## 
 ##    ##  ##  ##     ## ##    ## 
  ######  #### ##     ##  ######  
 
"""
def store_sims(runs=1000, simlen=50):
    F = np.array([[1, 0.5, 0,   0],
                  [0,   1, 0,   0],
                  [0,   0, 1, 0.5],
                  [0,   0, 0,   1]])
    Q = 0.001*np.array([[0.42, 1.25,    0,    0],
                        [1.25,    5,    0,    0],
                        [   0,    0, 0.42, 1.25],
                        [   0,    0, 1.25,    5]])
    H = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0]])
    Rs = [np.array([[ 4.77, -0.15],
                    [-0.15,  4.94]]),
          np.array([[ 2.99, -0.55],
                    [-0.55,  4.44]]),
          np.array([[2.06, 0.68],
                    [0.68, 1.96]]),
          np.array([[1.17,  0.8],
                    [ 0.8, 0.64]])]
    initgt = np.array([0, 1, 0, 1])
    initx = initgt
    initP = np.zeros((4,4))

    sims=[]
    # Sims
    for r in range(runs):
        # Reset sensors
        sensors = []
        for R in Rs:
            sensors.append(SimSensor(F, Q, H, R, initx, initP))
        
        # New enc keys
        pk, sk = paillier.generate_paillier_keypair(n_length=512)

        # Start sim
        gt = initgt
        sim = {'gt':[], 'fci':[], 'secfci':[], 'secfci2':[]}
        for k in range(simlen):
            gt = F@gt + np.random.multivariate_normal(np.zeros(4), Q)
            sim['gt'].append(gt)
            fci_data = []
            secfci_data = []
            secfci2_data = []
            for s in sensors:
                x, P = s.estimate(gt)
                tr = np.trace(P)
                Pinv = np.linalg.inv(P)
                Pinvx = Pinv@x
                
                # Sensor FCI
                fci_data.append((Pinvx, Pinv))
                
                # Sensor SecFCI
                enc_Pinv = np.array([[enc.EncryptedEncoding(pk, x) for x in row] for row in Pinv])
                enc_Pinvx = np.array([enc.EncryptedEncoding(pk, x) for x in Pinvx])
                # TODO Seem to have lost Python wrapper for ORE! Doesn't really matter since code is not timed and ORE adds no error
                # Pretend it's encrypted
                list_g01 = [tr*w for w in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]]
                list_g02 = [tr*w for w in [0,0.2,0.4,0.6,0.8,1]]
                list_g033 = [tr*w for w in [0,0.333,0.666,0.999,1]]
                list_g05 = [tr*w for w in [0,0.5,1]]
                secfci_data.append((enc_Pinvx, enc_Pinv, list_g01, list_g02, list_g033, list_g05))

                # Sensor SecFCI2
                enc_trinv = enc.EncryptedEncoding(pk, 1/tr)
                enc_trinvPinv = np.array([[enc.EncryptedEncoding(pk, x) for x in row] for row in (1/tr)*Pinv])
                enc_trinvPinvx = np.array([enc.EncryptedEncoding(pk, x) for x in (1/tr)*Pinvx])
                secfci2_data.append((enc_trinv, enc_trinvPinvx, enc_trinvPinv))

            # Cloud + Query FCI
            fci_Pinvx = np.zeros(4)
            fci_Pinv = np.zeros((4,4))
            fci_ws = []
            trinvsum = sum([1/np.trace(np.linalg.inv(d[1])) for d in fci_data])
            for d in fci_data:
                w = (1/np.trace(np.linalg.inv(d[1])))/trinvsum
                fci_Pinvx += w*d[0]
                fci_Pinv += w*d[1]
                fci_ws.append(w)
            fci_x = np.linalg.inv(fci_Pinv)@fci_Pinvx
            sim['fci'].append((fci_x, fci_ws))

            # Cloud + Query SecFCI

            # Cloud + Query SecFCI2
            enc_secfci2_trinv = sum([d[0] for d in secfci2_data])
            enc_secfci2_trinvPinvx = sum([d[1] for d in secfci2_data])
            enc_secfci2_trinvPinv = sum([d[2] for d in secfci2_data])
            secfci2_trinv = enc_secfci2_trinv.decrypt(sk)
            secfci2_trinvPinvx = np.array([x.decrypt(sk) for x in enc_secfci2_trinvPinvx])
            secfci2_trinvPinv = np.array([[x.decrypt(sk) for x in row] for row in enc_secfci2_trinvPinv])
            secfci2_Pinvx = (1/secfci2_trinv)*secfci2_trinvPinvx
            secfci2_Pinv = (1/secfci2_trinv)*secfci2_trinvPinv
            secfci2_x = np.linalg.inv(secfci2_Pinv)@secfci2_Pinvx
            sim['secfci2'].append(secfci2_x)


