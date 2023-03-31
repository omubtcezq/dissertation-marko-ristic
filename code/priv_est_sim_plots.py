"""

"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import plotting as pltng
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes


SIM_TIMESTEPS = 100
NUM_SIMS_TO_AVG = 1000
SAVE_FIGS = True
pltng.init_matplotlib_params(SAVE_FIGS, True)

"""
 
 ##     ## ######## ##       ########  ######## ########   ######  
 ##     ## ##       ##       ##     ## ##       ##     ## ##    ## 
 ##     ## ##       ##       ##     ## ##       ##     ## ##       
 ######### ######   ##       ########  ######   ########   ######  
 ##     ## ##       ##       ##        ##       ##   ##         ## 
 ##     ## ##       ##       ##        ##       ##    ##  ##    ## 
 ##     ## ######## ######## ##        ######## ##     ##  ######  
 
"""
# PLOTTING

def plot_avg_all_traces(plotter, state_covariances_lists, **kwargs):
    trace_lists = []
    for covs in state_covariances_lists:
        trace_lists.append([np.trace(P) for P in covs])
    mean_traces = np.mean(trace_lists, axis=0)
    return plotter.plot(range(len(mean_traces)), mean_traces, **kwargs)

def plot_avg_all_trace_diffs(plotter, state_covariances_lists1, state_covariances_lists2, **kwargs):
    trace_diff_lists = []
    for i in range(len(state_covariances_lists1)):
        covs1 = state_covariances_lists1[i]
        covs2 = state_covariances_lists2[i]
        trace_diff_lists.append([np.trace(P1)-np.trace(P2) for P1,P2 in zip(covs1, covs2)])
    mean_trace_diffs = np.mean(trace_diff_lists, axis=0)
    return plotter.plot(range(len(mean_trace_diffs)), mean_trace_diffs, **kwargs)

def plot_avg_all_sqr_error(plotter, states_lists, gts_lists, **kwargs):
    diff_lists = []
    for i in range(len(states_lists)):
        states = states_lists[i]
        gts = gts_lists[i]
        diff_lists.append([x@x for x in [s-g for s,g in zip(states,gts)]])
    mean_diffs = np.mean(diff_lists, axis=0)
    return plotter.plot(range(len(mean_diffs)), mean_diffs, **kwargs)

class GroundTruth:
    def __init__(self, F, Q, init_state):
        self.F = F
        self.Q = Q
        self.state = init_state
        return
    
    def update(self):
        w = np.random.multivariate_normal(np.array([0, 0, 0, 0]), self.Q)
        self.state = self.F@self.state + w
        return self.state

# ESTIMATION

class SensorAbs:
    def measure(self, ground_truth):
        raise NotImplementedError

class SensorPure(SensorAbs):
    def __init__(self, n, m, H, R):
        self.n = n
        self.m = m
        self.H = H
        self.R = R
        return
    
    def measure(self, ground_truth):
        v = np.random.multivariate_normal(np.array([0, 0]), self.R)
        return self.H@ground_truth + v

class SensorWithPrivileges(SensorPure):
    def __init__(self, n, m, H, R, covars_to_remove, generators):
        assert (len(covars_to_remove) == len(generators)), "Number of privilege classes does not match number of generators! Aborting."
        super().__init__(n, m, H, R)
        self.covars_to_remove = covars_to_remove
        self.generators = generators
        self.num_privs = len(covars_to_remove)
        return
    
    def measure(self, ground_truth):
        return super().measure(ground_truth) + self.get_sum_of_additional_noises()
    
    def get_sum_of_additional_noises(self):
        noise = 0
        for i in range(self.num_privs):
            n = self.generators[i].next_n_as_gaussian(self.m, np.array([0 for _ in range(self.m)]), self.covars_to_remove[i])
            noise += n
            #print("Sensor noise %d: " % i, n)
        #print("Sensor noise sum: ", noise)
        return noise

class FilterAbs:
    def predict(self):
        raise NotImplementedError
    
    def update(self, measurement):
        raise NotImplementedError

class KFilter(FilterAbs):
    def __init__(self, n, m, F, Q, H, R, init_state, init_cov):
        self.n = n
        self.m = m
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R
        self.x = init_state
        self.P = init_cov
        return
    
    def predict(self):
        self.x = self.F@self.x
        self.P = self.Q + (self.F@self.P@self.F.T)
        return self.x, self.P
    
    def update(self, measurement):
        S = (self.H@self.P@self.H.T) + self.R
        invS = np.linalg.inv(S)

        K = self.P@self.H.T@invS

        self.x = self.x + K@(measurement - self.H@self.x)
        self.P = self.P - (K@S@K.T)
        return self.x, self.P

class UnprivFilter(KFilter):
    def __init__(self, n, m, F, Q, H, R, init_state, init_cov, covars_to_remove):
        super().__init__(n, m, F, Q, H, R, init_state, init_cov)
        self.R = self.R + sum(covars_to_remove)
        return

class PrivFilter(KFilter):
    def __init__(self, n, m, F, Q, H, R, init_state, init_cov, priv_covar, covar_to_remove, generator):
        super().__init__(n, m, F, Q, H, R, init_state, init_cov)
        self.R = self.R + priv_covar
        self.covar_to_remove = covar_to_remove
        self.generator = generator
        return
    
    def update(self, measurement):
        super().update(measurement - self.get_additional_noise())
        return self.x, self.P
    
    def get_additional_noise(self):
        return self.generator.next_n_as_gaussian(self.m, np.array([0 for _ in range(self.m)]), self.covar_to_remove)

# KEY STREAMS

class KeyStreamPairFactory:
    @staticmethod
    def make_shared_key_streams(n):
        key = get_random_bytes(16)
        ciphers = []
        ciphers.append(AES.new(key, AES.MODE_CTR))
        for _ in range(n-1):
         ciphers.append(AES.new(key, AES.MODE_CTR, nonce=ciphers[0].nonce))
        return [KeyStream(cipher) for cipher in ciphers]


class KeyStream:
    def __init__(self, cipher):
        self.cipher = cipher
        self.bytes_to_read = 16
        self.max_read_int = 2**(16*8) - 1
        return
    
    def next(self):
        next_read = self.cipher.encrypt(self.bytes_to_read*b'\x00')
        next_int = int.from_bytes(next_read, byteorder='big', signed=False)
        return next_int
    
    def next_as_std_uniform(self):
        next_int = self.next()
        next_unif = (next_int/self.max_read_int)
        return next_unif
    
    def next_n_as_gaussian(self, n, mean, covariance):
        # Box-Muller transform for n standard normals from generated uniforms
        std_normals = []
        for i in range(n):
            if i%2 == 0:
                u1=self.next_as_std_uniform()
                u2=self.next_as_std_uniform()
                u1_cmp = np.sqrt(-2*np.log(u1))
                u2_cmp = 2*np.pi*u2
                std_normals.append(u1_cmp*np.cos(u2_cmp))
            else:
                std_normals.append(u1_cmp*np.sin(u2_cmp))
        std_normals = np.array(std_normals)
        # Conversion to samples from multivariate normal
        A = np.linalg.cholesky(covariance)
        return mean + A@std_normals

"""
 
  ######  #### ##     ##  ######  
 ##    ##  ##  ###   ### ##    ## 
 ##        ##  #### #### ##       
  ######   ##  ## ### ##  ######  
       ##  ##  ##     ##       ## 
 ##    ##  ##  ##     ## ##    ## 
  ######  #### ##     ##  ######  
 
"""

def save_sims():
    all_sims = {'B':{}, 'UB':{}}

    # State dimension
    n = 4
    # Measurement dimension
    m = 2
    # Process model (q = noise strength, t = timestep)
    q = 0.01
    t = 0.5
    F = np.array([[1, t, 0, 0], 
                  [0, 1, 0, 0], 
                  [0, 0, 1, t], 
                  [0, 0, 0, 1]])
    Q = q*np.array([[t**3/3, t**2/2,      0,      0], 
                    [t**2/2,      t,      0,      0], 
                    [     0,      0, t**3/3, t**2/2],
                    [     0,      0, t**2/2,      t]])
    # Measurement models
    H1 = np.array([[1, 0, 0, 0], 
                   [0, 0, 1, 0]])
    H2 = np.array([[0, 1, 0, 0], 
                   [0, 0, 0, 1]])
    R = np.array([[5, 2], 
                  [2, 5]])
    # Filter init
    init_state = np.array([0, 1, 0, 1])
    init_cov = np.array([[0, 0, 0, 0], 
                         [0, 0, 0, 0], 
                         [0, 0, 0, 0], 
                         [0, 0, 0, 0]])
    # Ground truth init
    gt_init_state = np.array([0, 1, 0, 1])

    # Single privilege class parameters
    covar_to_remove = np.array([[35, 0],
                                [0, 35]])
    # Encryption for single privilege class
    single_sensor_generator, single_filter_generator = KeyStreamPairFactory.make_shared_key_streams(2)

    """
    
    ########   #######  ##     ## ##    ## ########      ######  #### ##     ## 
    ##     ## ##     ## ##     ## ###   ## ##     ##    ##    ##  ##  ###   ### 
    ##     ## ##     ## ##     ## ####  ## ##     ##    ##        ##  #### #### 
    ########  ##     ## ##     ## ## ## ## ##     ##     ######   ##  ## ### ## 
    ##     ## ##     ## ##     ## ##  #### ##     ##          ##  ##  ##     ## 
    ##     ## ##     ## ##     ## ##   ### ##     ##    ##    ##  ##  ##     ## 
    ########   #######   #######  ##    ## ########      ######  #### ##     ## 
    
    """
    all_sim_gts = []
    all_sim_ys = []

    all_sim_unpriv_pred_lists = []
    all_sim_priv_pred_lists = []
    
    all_sim_unpriv_upd_lists = []
    all_sim_priv_upd_lists = []

    for i in range(NUM_SIMS_TO_AVG):
        print("Running bounded sim %d..." % (i+1))

        # Bounded filters
        unpriv_filter_bounded = UnprivFilter(n, m, F, Q, H1, R, init_state, init_cov, [covar_to_remove])
        priv_filter_bounded = PrivFilter(n, m, F, Q, H1, R, init_state, init_cov, np.zeros((2,2)), covar_to_remove, single_filter_generator)

        # Sensor
        sensor_bounded = SensorWithPrivileges(n, m, H1, R, [covar_to_remove], [single_sensor_generator])

        # Ground truth
        ground_truth = GroundTruth(F, Q, gt_init_state)

        gts = []
        ys = []

        unpriv_pred_list = []
        priv_pred_list = []

        unpriv_upd_list = []
        priv_upd_list = []

        # SIM LOOP
        for _ in range(SIM_TIMESTEPS):
            gt = ground_truth.update()
            y = sensor_bounded.measure(gt)

            # Predict
            unpriv_pred = unpriv_filter_bounded.predict()
            priv_pred = priv_filter_bounded.predict()
            
            # Update
            upriv_upd = unpriv_filter_bounded.update(y)
            priv_upd = priv_filter_bounded.update(y)
            
            # Save run data
            gts.append(gt)
            ys.append(y)

            unpriv_pred_list.append(unpriv_pred)
            priv_pred_list.append(priv_pred)
            
            unpriv_upd_list.append(upriv_upd)
            priv_upd_list.append(priv_upd)
        
        # SAVE ALL
        all_sim_gts.append(gts)
        all_sim_ys.append(ys)

        all_sim_unpriv_pred_lists.append(unpriv_pred_list)
        all_sim_priv_pred_lists.append(priv_pred_list)

        all_sim_unpriv_upd_lists.append(unpriv_upd_list)
        all_sim_priv_upd_lists.append(priv_upd_list)

    all_sims['B']['all_sim_gts'] = all_sim_gts
    all_sims['B']['all_sim_ys'] = all_sim_ys
    all_sims['B']['all_sim_unpriv_pred_lists'] = all_sim_unpriv_pred_lists
    all_sims['B']['all_sim_priv_pred_lists'] = all_sim_priv_pred_lists
    all_sims['B']['all_sim_unpriv_upd_lists'] = all_sim_unpriv_upd_lists
    all_sims['B']['all_sim_priv_upd_lists'] = all_sim_priv_upd_lists
    
    """
    
    ##     ## ##    ## ########   #######  ##     ## ##    ## ########      ######  #### ##     ## 
    ##     ## ###   ## ##     ## ##     ## ##     ## ###   ## ##     ##    ##    ##  ##  ###   ### 
    ##     ## ####  ## ##     ## ##     ## ##     ## ####  ## ##     ##    ##        ##  #### #### 
    ##     ## ## ## ## ########  ##     ## ##     ## ## ## ## ##     ##     ######   ##  ## ### ## 
    ##     ## ##  #### ##     ## ##     ## ##     ## ##  #### ##     ##          ##  ##  ##     ## 
    ##     ## ##   ### ##     ## ##     ## ##     ## ##   ### ##     ##    ##    ##  ##  ##     ## 
    #######  ##    ## ########   #######   #######  ##    ## ########      ######  #### ##     ## 
    
    """
    all_sim_gts = []
    all_sim_ys = []

    all_sim_unpriv_pred_lists = []
    all_sim_priv_pred_lists = []
    
    all_sim_unpriv_upd_lists = []
    all_sim_priv_upd_lists = []

    for i in range(NUM_SIMS_TO_AVG):
        print("Running unbounded sim %d..." % (i+1))

        # Unbounded filters
        unpriv_filter_unbounded = UnprivFilter(n, m, F, Q, H2, R, init_state, init_cov, [covar_to_remove])
        priv_filter_unbounded = PrivFilter(n, m, F, Q, H2, R, init_state, init_cov, np.zeros((2,2)), covar_to_remove, single_filter_generator)

        # Sensor
        sensor_unbounded = SensorWithPrivileges(n, m, H2, R, [covar_to_remove], [single_sensor_generator])

        # Ground truth
        ground_truth = GroundTruth(F, Q, gt_init_state)

        gts = []
        ys = []

        unpriv_pred_list = []
        priv_pred_list = []

        unpriv_upd_list = []
        priv_upd_list = []

        # SIM LOOP
        for _ in range(SIM_TIMESTEPS):
            gt = ground_truth.update()
            y = sensor_unbounded.measure(gt)

            # Predict
            unpriv_pred = unpriv_filter_unbounded.predict()
            priv_pred = priv_filter_unbounded.predict()
            
            # Update
            upriv_upd = unpriv_filter_unbounded.update(y)
            priv_upd = priv_filter_unbounded.update(y)
            
            # Save run data
            gts.append(gt)
            ys.append(y)

            unpriv_pred_list.append(unpriv_pred)
            priv_pred_list.append(priv_pred)
            
            unpriv_upd_list.append(upriv_upd)
            priv_upd_list.append(priv_upd)
        
        # SAVE ALL
        all_sim_gts.append(gts)
        all_sim_ys.append(ys)

        all_sim_unpriv_pred_lists.append(unpriv_pred_list)
        all_sim_priv_pred_lists.append(priv_pred_list)

        all_sim_unpriv_upd_lists.append(unpriv_upd_list)
        all_sim_priv_upd_lists.append(priv_upd_list)

    all_sims['UB']['all_sim_gts'] = all_sim_gts
    all_sims['UB']['all_sim_ys'] = all_sim_ys
    all_sims['UB']['all_sim_unpriv_pred_lists'] = all_sim_unpriv_pred_lists
    all_sims['UB']['all_sim_priv_pred_lists'] = all_sim_priv_pred_lists
    all_sims['UB']['all_sim_unpriv_upd_lists'] = all_sim_unpriv_upd_lists
    all_sims['UB']['all_sim_priv_upd_lists'] = all_sim_priv_upd_lists

    # Store to disk
    pickle.dump(all_sims, open("code/priv_estimation_sims.p", "wb"))


"""
 
 ########   #######  ##     ## ##    ## ########     ########  ##        #######  ######## 
 ##     ## ##     ## ##     ## ###   ## ##     ##    ##     ## ##       ##     ##    ##    
 ##     ## ##     ## ##     ## ####  ## ##     ##    ##     ## ##       ##     ##    ##    
 ########  ##     ## ##     ## ## ## ## ##     ##    ########  ##       ##     ##    ##    
 ##     ## ##     ## ##     ## ##  #### ##     ##    ##        ##       ##     ##    ##    
 ##     ## ##     ## ##     ## ##   ### ##     ##    ##        ##       ##     ##    ##    
 ########   #######   #######  ##    ## ########     ##        ########  #######     ##    
 
"""
def bound_plot():
    all_sims = pickle.load(open("code/priv_estimation_sims.p", "rb"))

    all_sim_gts = all_sims['B']['all_sim_gts']
    all_sim_ys = all_sims['B']['all_sim_ys']
    all_sim_unpriv_pred_lists = all_sims['B']['all_sim_unpriv_pred_lists']
    all_sim_priv_pred_lists = all_sims['B']['all_sim_priv_pred_lists']
    all_sim_unpriv_upd_lists = all_sims['B']['all_sim_unpriv_upd_lists']
    all_sim_priv_upd_lists = all_sims['B']['all_sim_priv_upd_lists']

    # Bounded plot
    print("Making single level bounded plot ...")

    fig_bounded = plt.figure()
    fig_bounded.set_size_inches(w=5.78853, h=0.75*0.7*5.78853)

    ax_tr_bounded = fig_bounded.add_subplot(121)
    ax_tr_bounded.set_ylabel(r"Trace")
    ax_tr_bounded.grid(True, c='lightgray')

    ax_mse_bounded = fig_bounded.add_subplot(122)
    ax_mse_bounded.set_ylabel(r"Mean Square Error (MSE)")
    ax_mse_bounded.grid(True, c='lightgray')

    fig_bounded.supxlabel(r'Simulation Timestep')

    # Size adjusting to fit in latex columns
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.75, wspace=0.4)

    diff_legend, = plot_avg_all_trace_diffs(ax_tr_bounded, [[s[1] for s in up_upd_l] for up_upd_l in all_sim_unpriv_upd_lists], 
                                                                      [[s[1] for s in pr_upd_l] for pr_upd_l in all_sim_priv_upd_lists], 
                                                                      linestyle=':', color='tab:orange')
    unpriv_legend, = plot_avg_all_traces(ax_tr_bounded, [[s[1] for s in up_upd_l] for up_upd_l in all_sim_unpriv_upd_lists], linestyle='-', color='tab:green')
    priv_legend, = plot_avg_all_traces(ax_tr_bounded, [[s[1] for s in pr_upd_l] for pr_upd_l in all_sim_priv_upd_lists], linestyle='-', color='tab:blue')
    

    plot_avg_all_sqr_error(ax_mse_bounded, [[s[0] for s in up_upd_l] for up_upd_l in all_sim_unpriv_upd_lists], all_sim_gts, linestyle='-', color='tab:green')
    plot_avg_all_sqr_error(ax_mse_bounded, [[s[0] for s in pr_upd_l] for pr_upd_l in all_sim_priv_upd_lists], all_sim_gts, linestyle='-', color='tab:blue')

    fig_bounded.legend(handles=[priv_legend, unpriv_legend, diff_legend], 
               labels=[r"Privileged Error ($\mat{P}^{(l)}$)", r"Unprivileged Error ($\mat{P}^{\prime(l)}$)", r"Difference ($\mat{D}$)"],
               loc="upper center",
               ncol=1)
    
    # Save or show figure
    if SAVE_FIGS:
        plt.savefig('figures/priv_estimation_est_sim_bounded.pdf')
    else:
        plt.show()

"""
 
 ##     ## ##    ## ########   #######  ##     ## ##    ## ########     ########  ##        #######  ######## 
 ##     ## ###   ## ##     ## ##     ## ##     ## ###   ## ##     ##    ##     ## ##       ##     ##    ##    
 ##     ## ####  ## ##     ## ##     ## ##     ## ####  ## ##     ##    ##     ## ##       ##     ##    ##    
 ##     ## ## ## ## ########  ##     ## ##     ## ## ## ## ##     ##    ########  ##       ##     ##    ##    
 ##     ## ##  #### ##     ## ##     ## ##     ## ##  #### ##     ##    ##        ##       ##     ##    ##    
 ##     ## ##   ### ##     ## ##     ## ##     ## ##   ### ##     ##    ##        ##       ##     ##    ##    
  #######  ##    ## ########   #######   #######  ##    ## ########     ##        ########  #######     ##    
 
"""
def unbound_plot():
    all_sims = pickle.load(open("code/priv_estimation_sims.p", "rb"))

    all_sim_gts = all_sims['UB']['all_sim_gts']
    all_sim_ys = all_sims['UB']['all_sim_ys']
    all_sim_unpriv_pred_lists = all_sims['UB']['all_sim_unpriv_pred_lists']
    all_sim_priv_pred_lists = all_sims['UB']['all_sim_priv_pred_lists']
    all_sim_unpriv_upd_lists = all_sims['UB']['all_sim_unpriv_upd_lists']
    all_sim_priv_upd_lists = all_sims['UB']['all_sim_priv_upd_lists']

    # Unbounded plot
    print("Making single level unbounded plot ...")

    fig_unbounded = plt.figure()
    fig_unbounded.set_size_inches(w=5.78853, h=0.75*0.7*5.78853)

    ax_tr_unbounded = fig_unbounded.add_subplot(121)
    ax_tr_unbounded.set_ylabel(r"Trace")
    ax_tr_unbounded.grid(True, c='lightgray')

    ax_mse_unbounded = fig_unbounded.add_subplot(122)
    ax_mse_unbounded.set_ylabel(r"Mean Square Error (MSE)")
    ax_mse_unbounded.grid(True, c='lightgray')

    # Size adjusting to fit in latex columns
    plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.75, wspace=0.4)

    diff_legend, = plot_avg_all_trace_diffs(ax_tr_unbounded, [[s[1] for s in up_upd_l] for up_upd_l in all_sim_unpriv_upd_lists], 
                                                                        [[s[1] for s in pr_upd_l] for pr_upd_l in all_sim_priv_upd_lists], 
                                                                        linestyle=':', color='tab:orange')
    unpriv_legend, = plot_avg_all_traces(ax_tr_unbounded, [[s[1] for s in up_upd_l] for up_upd_l in all_sim_unpriv_upd_lists], linestyle='-', color='tab:green')
    priv_legend, = plot_avg_all_traces(ax_tr_unbounded, [[s[1] for s in pr_upd_l] for pr_upd_l in all_sim_priv_upd_lists], linestyle='-', color='tab:blue')
    

    plot_avg_all_sqr_error(ax_mse_unbounded, [[s[0] for s in up_upd_l] for up_upd_l in all_sim_unpriv_upd_lists], all_sim_gts, linestyle='-', color='tab:green')
    plot_avg_all_sqr_error(ax_mse_unbounded, [[s[0] for s in pr_upd_l] for pr_upd_l in all_sim_priv_upd_lists], all_sim_gts, linestyle='-', color='tab:blue')

    fig_unbounded.legend(handles=[priv_legend, unpriv_legend, diff_legend], 
               labels=[r"Privileged Error ($\mat{P}^{(l)}$)", r"Unprivileged Error ($\mat{P}^{\prime(l)}$)", r"Difference ($\mat{D}$)"],
               loc="upper center",
               ncol=1)
    
    # Save or show figure
    if SAVE_FIGS:
        plt.savefig('figures/priv_estimation_est_sim_unbounded.pdf')
    else:
        plt.show()



#save_sims()
bound_plot()
unbound_plot()