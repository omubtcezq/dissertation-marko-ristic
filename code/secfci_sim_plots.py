import numpy as np
from phe import paillier
import pickle
import matplotlib.pyplot as plt

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
        z = self.H@gt + np.random.multivariate_normal(np.zeros(2), self.R)
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

# SecFCI 2 sensor list intersection approx function (l2 should already be reversed)
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

# SecFCI n sensor linear system functions
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

def omegas_from_ore_lists(sensor_lists, stepsize):
    n = len(sensor_lists)

    # Linear system dimensions
    res = np.zeros(n*(n-1))
    mat = np.zeros((n*(n-1),n+(n-1)*(n-1)))

    # Get computable point, trivial points and differences
    for i in range(n-1):
        omega = intersect_approx_bsearch(sensor_lists[i], sorted(sensor_lists[i+1], reverse=True), stepsize)
        assert(0<=omega<=1)
        assert((list(np.array(sensor_lists[i]) < np.array(sorted(sensor_lists[i+1], reverse=True))).index(False)-0.5)*stepsize == omega)
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
    sol_vec, err, rnk, sing = np.linalg.lstsq(mat, res, rcond=None)
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
    # initx = initgt + np.random.multivariate_normal(np.zeros(4), 5*np.eye(4))
    # initP = 5*np.eye(4)

    sims=[]
    # Sims
    for r in range(runs):
        # Progress
        print("Running sim %d / %d" % (r+1, runs))

        # Reset sensors
        sensors = []
        for R in Rs:
            sensors.append(SimSensor(F, Q, H, R, initx, initP))
        
        # New enc keys
        pk, sk = paillier.generate_paillier_keypair(n_length=512)

        # Start sim
        gt = initgt
        sim = {'gt':[], 'fci':[], 'secfci':[], 'secfci2':[]}
        for _ in range(simlen):
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
                list_g01 = [tr*w for w in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]]
                list_g02 = [tr*w for w in [0, 0.2, 0.4, 0.6, 0.8, 1]]
                list_g033 = [tr*w for w in [0, 0.33, 0.66, 1]]
                list_g05 = [tr*w for w in [0, 0.5, 1]]
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
            g01_ws = omegas_from_ore_lists([x[2] for x in secfci_data], 0.1)
            enc_secfci_Pinvx_g01 = sum([w*Pinvx for w,Pinvx in zip(g01_ws, [d[0] for d in secfci_data])])
            enc_secfci_Pinv_g01 = sum([w*Pinv for w,Pinv in zip(g01_ws, [d[1] for d in secfci_data])])
            secfci_Pinvx_g01 = np.array([x.decrypt(sk) for x in enc_secfci_Pinvx_g01])
            secfci_Pinv_g01 = np.array([[x.decrypt(sk) for x in row] for row in enc_secfci_Pinv_g01])
            secfci_x_g01 = np.linalg.inv(secfci_Pinv_g01)@secfci_Pinvx_g01

            g02_ws = omegas_from_ore_lists([x[3] for x in secfci_data], 0.2)
            enc_secfci_Pinvx_g02 = sum([w*Pinvx for w,Pinvx in zip(g02_ws, [d[0] for d in secfci_data])])
            enc_secfci_Pinv_g02 = sum([w*Pinv for w,Pinv in zip(g02_ws, [d[1] for d in secfci_data])])
            secfci_Pinvx_g02 = np.array([x.decrypt(sk) for x in enc_secfci_Pinvx_g02])
            secfci_Pinv_g02 = np.array([[x.decrypt(sk) for x in row] for row in enc_secfci_Pinv_g02])
            secfci_x_g02 = np.linalg.inv(secfci_Pinv_g02)@secfci_Pinvx_g02

            g033_ws = omegas_from_ore_lists([x[4] for x in secfci_data], 0.33)
            enc_secfci_Pinvx_g033 = sum([w*Pinvx for w,Pinvx in zip(g033_ws, [d[0] for d in secfci_data])])
            enc_secfci_Pinv_g033 = sum([w*Pinv for w,Pinv in zip(g033_ws, [d[1] for d in secfci_data])])
            secfci_Pinvx_g033 = np.array([x.decrypt(sk) for x in enc_secfci_Pinvx_g033])
            secfci_Pinv_g033 = np.array([[x.decrypt(sk) for x in row] for row in enc_secfci_Pinv_g033])
            secfci_x_g033 = np.linalg.inv(secfci_Pinv_g033)@secfci_Pinvx_g033

            g05_ws = omegas_from_ore_lists([x[5] for x in secfci_data], 0.5)
            enc_secfci_Pinvx_g05 = sum([w*Pinvx for w,Pinvx in zip(g05_ws, [d[0] for d in secfci_data])])
            enc_secfci_Pinv_g05 = sum([w*Pinv for w,Pinv in zip(g05_ws, [d[1] for d in secfci_data])])
            secfci_Pinvx_g05 = np.array([x.decrypt(sk) for x in enc_secfci_Pinvx_g05])
            secfci_Pinv_g05 = np.array([[x.decrypt(sk) for x in row] for row in enc_secfci_Pinv_g05])
            secfci_x_g05 = np.linalg.inv(secfci_Pinv_g05)@secfci_Pinvx_g05

            sim['secfci'].append(((secfci_x_g01, secfci_x_g02, secfci_x_g033, secfci_x_g05), (g01_ws, g02_ws, g033_ws, g05_ws)))


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

        sims.append(sim)
    
    # Store to disk
    pickle.dump(sims, open("code/cloud_fusion_sims.p", "wb"))

"""
 
    ##      ######## ########  ########     ########  ##        #######  ######## 
  ####      ##       ##     ## ##     ##    ##     ## ##       ##     ##    ##    
    ##      ##       ##     ## ##     ##    ##     ## ##       ##     ##    ##    
    ##      ######   ########  ########     ########  ##       ##     ##    ##    
    ##      ##       ##   ##   ##   ##      ##        ##       ##     ##    ##    
    ##      ##       ##    ##  ##    ##     ##        ##       ##     ##    ##    
  ######    ######## ##     ## ##     ##    ##        ########  #######     ##    
 
"""
def secfci_error_plot():
    fig = plt.figure()
    # 70% text width and default matplotlib aspect ratio
    fig.set_size_inches(w=0.7*5.78853, h=0.75*0.7*5.78853)
    ax = fig.add_subplot(111)
    ax.grid(True, c='lightgray')

    sims = pickle.load(open("code/cloud_fusion_sims.p", "rb"))
    ks = [k for k in range(len(sims[0]['gt']))]

    fci_errs = [np.mean([sum((sim['gt'][k] - sim['fci'][k][0])**2) for sim in sims]) for k in ks]
    g01_errs = [np.mean([sum((sim['gt'][k] - sim['secfci'][k][0][0])**2) for sim in sims]) for k in ks]
    g02_errs = [np.mean([sum((sim['gt'][k] - sim['secfci'][k][0][1])**2) for sim in sims]) for k in ks]
    g033_errs = [np.mean([sum((sim['gt'][k] - sim['secfci'][k][0][2])**2) for sim in sims]) for k in ks]
    g05_errs = [np.mean([sum((sim['gt'][k] - sim['secfci'][k][0][3])**2) for sim in sims]) for k in ks]

    ax.plot(ks, fci_errs, marker='', label=r'FCI Benchmark')
    ax.plot(ks, g05_errs, marker='', label=r'$g=0.5$')
    ax.plot(ks, g033_errs, marker='', label=r'$g=0.33$')
    ax.plot(ks, g02_errs, marker='', label=r'$g=0.2$')
    ax.plot(ks, g01_errs, marker='', label=r'$g=0.1$')
    
    ax.set_xlabel(r'Simulation Timestep')
    ax.set_ylabel(r'Mean Square Error (MSE)')

    plt.legend()
    #ax.set_yticks([trA, trB])
    #ax.set_yticklabels([r'$\tr(\mat{P}_1)$', r'$\tr(\mat{P}_2)$'])
    #ax.set_xticks(list(np.arange(0,1.1,0.1))+[0.5*(l+r)])
    #ax.set_xticklabels([str(x/10) if x in [0,5,10] else '' for x in range(11)]+[r'$\hat{\omega}_1$'])

    if SAVE_FIGS:
        plt.savefig('figures/cloud_fusion_secfci_sim_error.pdf')
    else:
        plt.show()
    plt.close()
    return

"""
 
  ######      ########  ##        #######  ######## 
 ##    ##     ##     ## ##       ##     ##    ##    
 ##           ##     ## ##       ##     ##    ##    
 ##   ####    ########  ##       ##     ##    ##    
 ##    ##     ##        ##       ##     ##    ##    
 ##    ##     ##        ##       ##     ##    ##    
  ######      ##        ########  #######     ##    
 
"""
def secfci_omega_plot():
    fig = plt.figure()
    # 70% text width and default matplotlib aspect ratio
    fig.set_size_inches(w=0.7*5.78853, h=0.75*0.7*5.78853)
    ax = fig.add_subplot(111)
    ax.grid(True, c='lightgray')

    sims = pickle.load(open("code/cloud_fusion_sims.p", "rb"))

    w_errs = [sum((np.array(sims[0]['fci'][-1][1]) - np.array(sims[0]['secfci'][-1][1][g]))**2) for g in range(4)]

    ax.plot([0.1,0.2,0.33,0.5], w_errs, marker='', label=r'$\hat{\vec{\omega}}$ Estimate Error')

    ax.set_xlabel(r'$g$')
    ax.set_ylabel(r'Mean Square Error (MSE)')

    plt.legend()
    #ax.set_yticks([trA, trB])
    #ax.set_yticklabels([r'$\tr(\mat{P}_1)$', r'$\tr(\mat{P}_2)$'])
    ax.set_xticks([0.1,0.2,0.33,0.5])
    #ax.set_xticklabels([str(x/10) if x in [0,5,10] else '' for x in range(11)]+[r'$\hat{\omega}_1$'])

    if SAVE_FIGS:
        plt.savefig('figures/cloud_fusion_secfci_omega_error.pdf')
    else:
        plt.show()
    plt.close()
    return

"""
 
  #######     ######## ########  ########     ########  ##        #######  ######## 
 ##     ##    ##       ##     ## ##     ##    ##     ## ##       ##     ##    ##    
        ##    ##       ##     ## ##     ##    ##     ## ##       ##     ##    ##    
  #######     ######   ########  ########     ########  ##       ##     ##    ##    
 ##           ##       ##   ##   ##   ##      ##        ##       ##     ##    ##    
 ##           ##       ##    ##  ##    ##     ##        ##       ##     ##    ##    
 #########    ######## ##     ## ##     ##    ##        ########  #######     ##    
 
"""
def secfci2_error_plot():
    fig = plt.figure()
    # 70% text width and default matplotlib aspect ratio
    fig.set_size_inches(w=0.7*5.78853, h=0.75*0.7*5.78853)
    ax = fig.add_subplot(111)
    ax.grid(True, c='lightgray')

    sims = pickle.load(open("code/cloud_fusion_sims.p", "rb"))
    ks = [k for k in range(len(sims[0]['gt']))]

    fci_errs = [np.mean([sum((sim['gt'][k] - sim['fci'][k][0])**2) for sim in sims]) for k in ks]
    secfci_errs = [np.mean([sum((sim['gt'][k] - sim['secfci'][k][0][1])**2) for sim in sims]) for k in ks]
    secfci2_errs = [np.mean([sum((sim['gt'][k] - sim['secfci2'][k])**2) for sim in sims]) for k in ks]

    ax.plot(ks, fci_errs, marker='', label=r'FCI Benchmark')
    ax.plot(ks, secfci_errs, marker='', label=r'Leaking Weights ($g=0.2$)')
    ax.plot(ks, secfci2_errs, marker='', linestyle='--', label=r'Without Leaking Weights')

    ax.set_xlabel(r'Simulation Timestep')
    ax.set_ylabel(r'Mean Square Error (MSE)')

    plt.legend()
    #ax.set_yticks([trA, trB])
    #ax.set_yticklabels([r'$\tr(\mat{P}_1)$', r'$\tr(\mat{P}_2)$'])
    #ax.set_xticks(list(np.arange(0,1.1,0.1))+[0.5*(l+r)])
    #ax.set_xticklabels([str(x/10) if x in [0,5,10] else '' for x in range(11)]+[r'$\hat{\omega}_1$'])

    if SAVE_FIGS:
        plt.savefig('figures/cloud_fusion_secfci2_sim_error.pdf')
    else:
        plt.show()
    plt.close()
    return


store_sims(runs=10)
#secfci_error_plot()
#secfci_omega_plot()
#secfci2_error_plot()