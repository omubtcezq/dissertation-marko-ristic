
import numpy as np
import pickle
import matplotlib.pyplot as plt
import plotting as pltng
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes


SIM_RUNS = 1000
SIM_STEPS = 100
PROGRESS_PRINTS = 25

DO_PRIV_SIM = True
DO_PARAM_SIM = True
DO_SCAN_SIM = True

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
class PrivilegeSimData:
    def __init__(self, ident, num_sensors):
        # Store general simulation information
        self.ident = ident
        self.num_sensors = num_sensors
        self.gt = []
        self.zs = dict(((s, []) for s in range(num_sensors)))
        # Results from unprivileged and privileged filters for all the privileges considered
        self.unpriv_filter_results = []
        self.priv_filters_j_ms_results = dict(((p, []) for p in range(num_sensors)))
        self.priv_filters_all_ms_results = dict(((p, []) for p in range(num_sensors)))
        return
    
    def compute_errors(self):
        self.sim_len = len(self.gt)
        
        # Compute unprivileged errors
        self.unpriv_filter_errors = [np.linalg.norm(self.unpriv_filter_results[i][0] - self.gt[i])**2 for i in range(self.sim_len)]

        # Compute privileges errors for all privileges considered
        self.priv_filters_j_ms_errors = {}
        self.priv_filters_all_ms_errors = {}
        for p in range(self.num_sensors):
            self.priv_filters_j_ms_errors[p] = [np.linalg.norm(self.priv_filters_j_ms_results[p][i][0] - self.gt[i])**2 for i in range(self.sim_len)]
            self.priv_filters_all_ms_errors[p] = [np.linalg.norm(self.priv_filters_all_ms_results[p][i][0] - self.gt[i])**2 for i in range(self.sim_len)]

        return

class AvgPrivilegeSimData:
    def __init__(self, sim_list):
        # Copy general information from the first simulation
        self.num_sensors = sim_list[0].num_sensors
        self.sim_len = sim_list[0].sim_len

        # Save first simulation to be able to plot covariance trace as a comparison (doesn't matter which sim is actually saved)
        self.last_sim = sim_list[0]

        # Average the results of unprivileged and privileged filters for all the privileges considered
        self.unpriv_filter_errors_avg = np.mean([s.unpriv_filter_errors for s in sim_list], axis=0)
        self.priv_filters_j_ms_errors_avg = {}
        self.priv_filters_all_ms_errors_avg = {}
        for p in range(self.num_sensors):
            self.priv_filters_j_ms_errors_avg[p] = np.mean([s.priv_filters_j_ms_errors[p] for s in sim_list], axis=0)
            self.priv_filters_all_ms_errors_avg[p] = np.mean([s.priv_filters_all_ms_errors[p] for s in sim_list], axis=0)
        return


class ParameterSimData:
    def __init__(self, ident, num_sensors, Ys, Zs):
        # Store general simulation information
        self.ident = ident
        self.num_sensors = num_sensors
        self.gt = []

        # Store the parameter values that will be varied
        self.Ys = Ys
        self.Zs = Zs

        # As params are varied, store all measurements, unprivielged and privileged filters' estimates in a 2-D dictionary of parameters
        self.zs = {}
        self.unpriv_filters_results = {}
        self.priv_filters_j_ms_results = {}
        self.priv_filters_all_ms_results = {}
        for Y in Ys:
            self.zs[Y] = {}
            self.unpriv_filters_results[Y] = {}
            self.priv_filters_j_ms_results[Y] = {}
            self.priv_filters_all_ms_results[Y] = {}
            for Z in Zs:
                self.zs[Y][Z] = dict(((s, []) for s in range(num_sensors)))
                self.unpriv_filters_results[Y][Z] = []
                self.priv_filters_j_ms_results[Y][Z] = []
                self.priv_filters_all_ms_results[Y][Z] = []
        
        return
    
    def compute_errors(self):
        self.sim_len = len(self.gt)

        # Compute errors of unprivileged, privileged with denoised measurements only and privileged with all measuremets filters, for all parameter combinations
        self.unpriv_filters_errors = {}
        self.priv_filters_j_ms_errors = {}
        self.priv_filters_all_ms_errors = {}
        for Y in self.Ys:
            self.unpriv_filters_errors[Y] = {}
            self.priv_filters_j_ms_errors[Y] = {}
            self.priv_filters_all_ms_errors[Y] = {}
            for Z in self.Zs:
                self.unpriv_filters_errors[Y][Z] = [np.linalg.norm(self.unpriv_filters_results[Y][Z][i][0] - self.gt[i])**2 for i in range(self.sim_len)]
                self.priv_filters_j_ms_errors[Y][Z] = [np.linalg.norm(self.priv_filters_j_ms_results[Y][Z][i][0] - self.gt[i])**2 for i in range(self.sim_len)]
                self.priv_filters_all_ms_errors[Y][Z] = [np.linalg.norm(self.priv_filters_all_ms_results[Y][Z][i][0] - self.gt[i])**2 for i in range(self.sim_len)]

        return

class AvgParameterSimData:
    def __init__(self, sim_list):
        # Copy general information from the first simulation
        self.num_sensors = sim_list[0].num_sensors
        self.sim_len = sim_list[0].sim_len
        self.Ys = sim_list[0].Ys
        self.Zs = sim_list[0].Zs

        # Save first simulation to be able to plot covariance trace as a comparison (doesn't matter which sim is actually saved)
        self.last_sim = sim_list[0]

        # Compute average errors of all considered filters for all parameter combinations
        self.unpriv_filters_errors_avg = {}
        self.priv_filters_j_ms_errors_avg = {}
        self.priv_filters_all_ms_errors_avg = {}
        for Y in self.Ys:
            self.unpriv_filters_errors_avg[Y] = {}
            self.priv_filters_j_ms_errors_avg[Y] = {}
            self.priv_filters_all_ms_errors_avg[Y] = {}
            for Z in self.Zs:
                self.unpriv_filters_errors_avg[Y][Z] = np.mean([s.unpriv_filters_errors[Y][Z] for s in sim_list], axis=0)
                self.priv_filters_j_ms_errors_avg[Y][Z] = np.mean([s.priv_filters_j_ms_errors[Y][Z] for s in sim_list], axis=0)
                self.priv_filters_all_ms_errors_avg[Y][Z] = np.mean([s.priv_filters_all_ms_errors[Y][Z] for s in sim_list], axis=0)
        return

class ParameterScanSimData:
    def __init__(self, ident, num_sensors, Y_fixed, Z_fixed, Ys, Zs, privileges):
        # Store general simulation information
        self.ident = ident
        self.num_sensors = num_sensors
        self.gt = []

        # Store the values that will be varied
        self.Y_fixed = Y_fixed
        self.Z_fixed = Z_fixed
        self.Ys = Ys
        self.Zs = Zs
        self.privileges = privileges

        # As params are varied, store all measurements, unprivielged and privileged filters' estimates in a 2-D dictionary of parameters
        self.zs = {}
        self.unpriv_filters_results = {}
        self.priv_filters_j_ms_results = {}
        self.priv_filters_all_ms_results = {}
        for priv in privileges:
            self.zs[priv] = dict((("Z_fixed", {}), ("Y_fixed", {})))
            self.unpriv_filters_results[priv] = dict((("Z_fixed", {}), ("Y_fixed", {})))
            self.priv_filters_j_ms_results[priv] = dict((("Z_fixed", {}), ("Y_fixed", {})))
            self.priv_filters_all_ms_results[priv] = dict((("Z_fixed", {}), ("Y_fixed", {})))
            for Y in Ys:
                self.zs[priv]["Z_fixed"][Y] = dict(((s, []) for s in range(num_sensors)))
                self.unpriv_filters_results[priv]["Z_fixed"][Y] = []
                self.priv_filters_j_ms_results[priv]["Z_fixed"][Y] = []
                self.priv_filters_all_ms_results[priv]["Z_fixed"][Y] = []
            for Z in Zs:
                self.zs[priv]["Y_fixed"][Z] = dict(((s, []) for s in range(num_sensors)))
                self.unpriv_filters_results[priv]["Y_fixed"][Z] = []
                self.priv_filters_j_ms_results[priv]["Y_fixed"][Z] = []
                self.priv_filters_all_ms_results[priv]["Y_fixed"][Z] = []
        
        return
    
    def compute_steady_state_traces(self):
        self.sim_len = len(self.gt)

        # Compute errors of unprivileged, privileged with denoised measurements only and privileged with all measuremets filters, for all parameter combinations
        self.unpriv_filters_traces = {}
        self.priv_filters_j_ms_traces = {}
        self.priv_filters_all_ms_traces = {}
        for priv in self.privileges:
            self.unpriv_filters_traces[priv] = {}
            self.priv_filters_j_ms_traces[priv] = {}
            self.priv_filters_all_ms_traces[priv] = {}
            
            self.unpriv_filters_traces[priv]["Z_fixed"] = [np.trace(self.unpriv_filters_results[priv]["Z_fixed"][Y][-1][1]) for Y in self.Ys]
            self.priv_filters_j_ms_traces[priv]["Z_fixed"] = [np.trace(self.priv_filters_j_ms_results[priv]["Z_fixed"][Y][-1][1]) for Y in self.Ys]
            self.priv_filters_all_ms_traces[priv]["Z_fixed"] = [np.trace(self.priv_filters_all_ms_results[priv]["Z_fixed"][Y][-1][1]) for Y in self.Ys]
            
            self.unpriv_filters_traces[priv]["Y_fixed"] = [np.trace(self.unpriv_filters_results[priv]["Y_fixed"][Z][-1][1]) for Z in self.Zs]
            self.priv_filters_j_ms_traces[priv]["Y_fixed"] = [np.trace(self.priv_filters_j_ms_results[priv]["Y_fixed"][Z][-1][1]) for Z in self.Zs]
            self.priv_filters_all_ms_traces[priv]["Y_fixed"] = [np.trace(self.priv_filters_all_ms_results[priv]["Y_fixed"][Z][-1][1]) for Z in self.Zs]
        return
    
# ESTIMATION
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

class PrivFusionFilter(KFilter):
    def __init__(self, n, m, F, Q, H, R, init_state, init_cov, Z, Y, generators, num_measurements):
        self.Z = Z
        self.Y = Y
        self.generators = generators
        self.num_measurements = num_measurements

        self.single_m = m
        self.privilege = len(generators)
        self.correlated_noise_covariance = None
        if self.privilege > 0:
            self.correlated_noise_covariance = np.block([[Z+Y if c==r else Z for c in range(self.privilege)] for r in range(self.privilege)])

        stacked_m = m*num_measurements
        stacked_H = np.block([[H] for _ in range(num_measurements)])

        self.num_unpriv_measurements = self.num_measurements - self.privilege

        # Unprivileged
        if self.num_unpriv_measurements == self.num_measurements:
            stacked_R = np.block([[R+Z+Y if c==r else Z for c in range(self.num_measurements)] for r in range(self.num_measurements)])
        
        # Privileged
        elif self.num_unpriv_measurements == 0:
            stacked_R = np.block([[R if c==r else np.zeros((2,2)) for c in range(num_measurements)] for r in range(num_measurements)])
        
        # Privileged with additional measurements
        else:
            priv_cov = np.block([[R if c==r else np.zeros((2,2)) for c in range(self.privilege)] for r in range(self.privilege)])

            known_noise_cov = np.block([[Z+Y if c==r else Z for c in range(self.privilege)] for r in range(self.privilege)])
            unknown_noise_cov = np.block([[Z+Y if c==r else Z for c in range(self.num_unpriv_measurements)] for r in range(self.num_unpriv_measurements)])
            known_unknown_cross_cov = np.block([[Z for _ in range(self.num_unpriv_measurements)] for _ in range(self.privilege)])

            self.unknown_additive_noise_offset = known_unknown_cross_cov.T@np.linalg.inv(known_noise_cov)

            unpriv_cov = unknown_noise_cov - known_unknown_cross_cov.T@np.linalg.inv(known_noise_cov)@known_unknown_cross_cov
            unpriv_cov = unpriv_cov + np.block([[R if c==r else np.zeros((2,2)) for c in range(self.num_unpriv_measurements)] for r in range(self.num_unpriv_measurements)])

            independent_cross_cov = np.block([[np.zeros((2,2)) for _ in range(self.num_unpriv_measurements)] for _ in range(self.privilege)])
            stacked_R = np.block([[priv_cov, independent_cross_cov],[independent_cross_cov.T, unpriv_cov]])

        super().__init__(n, stacked_m, F, Q, stacked_H, stacked_R, init_state, init_cov)
        return
    
    def update(self, measurements):
        # Only generate noises if holding any keys
        if self.privilege > 0:

            # Generate the known noises
            std_normals = np.block([g.next_n_as_std_gaussian(self.single_m) for g in self.generators])
            correlated_noises = np.linalg.cholesky(self.correlated_noise_covariance)@std_normals

            if self.num_unpriv_measurements > 0:
                padding = self.unknown_additive_noise_offset@correlated_noises
                correlated_noises = np.block([correlated_noises, padding])

            # Remove the noises from the recieved measurements
            measurements = measurements - correlated_noises

        # Run filter udpate
        super().update(measurements)
        return self.x, self.P

# KEYSTREAMS
class SharedKeyStreamFactory:
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
    
    def next_n_as_std_gaussian(self, n):
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
        return np.array(std_normals)

    def next_n_as_gaussian(self, n, mean, covariance):
        std_normals = self.next_n_as_std_gaussian(n)
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
def run_simulations():
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
    H = np.array([[1, 0, 0, 0], 
                  [0, 0, 1, 0]])

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
    # Number of present sensors
    num_sensors = 4

    # Data to store and save
    all_sims = {}

    """
    
    ########  ########  #### ##     ##  ######  
    ##     ## ##     ##  ##  ##     ## ##    ## 
    ##     ## ##     ##  ##  ##     ## ##       
    ########  ########   ##  ##     ##  ######  
    ##        ##   ##    ##   ##   ##        ## 
    ##        ##    ##   ##    ## ##   ##    ## 
    ##        ##     ## ####    ###     ######  
    
    """
    if DO_PRIV_SIM:
        sims = []
        print("\nMaking Privilege Plot ...\n")
        for s in range(SIM_RUNS):
            # Progress printing
            if s % PROGRESS_PRINTS == 0:
                print("Running Simulation %d ..." % s)
            
            # Pseudorandom correleated and uncorrelated covariances
            Z = 2*np.eye(2)
            Y = 10*np.eye(2)
            sensor_correlated_covariance = np.block([[Z+Y if r==c else Z for c in range(num_sensors)] for r in range(num_sensors)])
            
            # Sim data storage
            sim = PrivilegeSimData(s, num_sensors)
            sims.append(sim)

            # Synced cryptographically random number generators. Makes exactly the amount of each required by the simulation.
            # Remade for each individual simulation to make popping from generator lists easier
            sensor_generators = [SharedKeyStreamFactory.make_shared_key_streams(1+2*num_sensors-2*s) for s in range(num_sensors)]

            # Creating simulation objects (ground truth, sensors and filters)
            ground_truth = GroundTruth(F, Q, gt_init_state)
            sensors = []
            for _ in range(num_sensors):
                sensors.append(SensorPure(n, m, H, R))

            unpriv_filter = PrivFusionFilter(n, m, F, Q, H, R, init_state, init_cov, Z, Y, [], num_sensors)
            priv_filters_j_ms = []
            priv_filters_all_ms = []
            for j in range(num_sensors):
                gens = [g.pop() for g in sensor_generators[:j+1]]
                priv_filters_j_ms.append(PrivFusionFilter(n, m, F, Q, H, R, init_state, init_cov, Z, Y, gens, j+1))

                gens = [g.pop() for g in sensor_generators[:j+1]]
                priv_filters_all_ms.append(PrivFusionFilter(n, m, F, Q, H, R, init_state, init_cov, Z, Y, gens, num_sensors))

            # Run simulation
            for _ in range(SIM_STEPS):
                
                # Update ground truth
                gt = ground_truth.update()
                sim.gt.append(gt)

                # Generate correlated noise
                std_normals = np.block([g[0].next_n_as_std_gaussian(m) for g in sensor_generators])
                correlated_noises = np.linalg.cholesky(sensor_correlated_covariance)@std_normals

                # Make all measurements and add pseudorandom noises
                zs = []
                for sen in range(num_sensors):
                    true_z = sensors[sen].measure(gt)
                    z = true_z + correlated_noises[sen*m:sen*m+m]
                    sim.zs[sen].append(z)
                    zs.append(z)

                # Unprivileged filter estimate
                unpriv_filter.predict()
                res = unpriv_filter.update(np.block(zs))
                sim.unpriv_filter_results.append(res)

                # Privileged with denoised measurements only and with all measurements filters' estimates
                for j in range(num_sensors):
                    priv_filters_j_ms[j].predict()
                    res_j = priv_filters_j_ms[j].update(np.block(zs[:j+1]))
                    sim.priv_filters_j_ms_results[j].append(res_j)

                    priv_filters_all_ms[j].predict()
                    res_all = priv_filters_all_ms[j].update(np.block(zs))
                    sim.priv_filters_all_ms_results[j].append(res_all)
        
            # Compute errors of the filters
            sim.compute_errors()

        # Average simulations
        avg_sim_data = AvgPrivilegeSimData(sims)

        # Save sims
        all_sims['privs'] = avg_sim_data

    """
    
    ########     ###    ########     ###    ##     ##  ######  
    ##     ##   ## ##   ##     ##   ## ##   ###   ### ##    ## 
    ##     ##  ##   ##  ##     ##  ##   ##  #### #### ##       
    ########  ##     ## ########  ##     ## ## ### ##  ######  
    ##        ######### ##   ##   ######### ##     ##       ## 
    ##        ##     ## ##    ##  ##     ## ##     ## ##    ## 
    ##        ##     ## ##     ## ##     ## ##     ##  ######  
    
    """
    if DO_PARAM_SIM:
        sims = []
        print("\nMaking Parameter Plot ...\n")
        for s in range(SIM_RUNS):
            # Progress printing
            if s % PROGRESS_PRINTS == 0:
                print("Running Simulation %d ..." % s)
            
            # Varying pseudorandom correlated and uncorrelated covariances
            Ys = [2, 10]
            Zs = [2, 10]

            # Only one privilege (number of keys) considered in this plot
            fixed_privilege = 2

            # Sim data storage
            sim = ParameterSimData(s, num_sensors, Ys, Zs)
            sims.append(sim)

            # Synced cryptographically random number generators. Makes exactly the amount of each required by the simulation.
            # Remade for each individual simulation to make popping from generator lists easier
            sensor_generators = [SharedKeyStreamFactory.make_shared_key_streams(9-4*(s-(s%2))) for s in range(num_sensors)]

            # Creating simulation objects (ground truth, sensors and filters)
            ground_truth = GroundTruth(F, Q, gt_init_state)
            sensors = []
            for _ in range(num_sensors):
                sensors.append(SensorPure(n, m, H, R))

            # As correlation parameters are being changed, store all filters and correlation matrices in 2-D dictionaries
            sensor_correlated_covariances = {}
            unpriv_filters = {}
            priv_filters_j_ms = {}
            priv_filters_all_ms = {}
            for Y in Ys:
                sensor_correlated_covariances[Y] = {}
                unpriv_filters[Y] = {}
                priv_filters_j_ms[Y] = {}
                priv_filters_all_ms[Y] = {}
                for Z in Zs:
                    # Correlation matrix
                    Y_mat = Y*np.eye(2)
                    Z_mat = Z*np.eye(2)
                    sensor_correlated_covariances[Y][Z] = np.block([[Z_mat+Y_mat if r==c else Z_mat for c in range(num_sensors)] for r in range(num_sensors)])

                    # Unpriv filter
                    unpriv_filters[Y][Z] = PrivFusionFilter(n, m, F, Q, H, R, init_state, init_cov, Z_mat, Y_mat, [], num_sensors)

                    # Priv filter, denoisable measurements only
                    gens = [g.pop() for g in sensor_generators[:fixed_privilege]]
                    priv_filters_j_ms[Y][Z] = PrivFusionFilter(n, m, F, Q, H, R, init_state, init_cov, Z_mat, Y_mat, gens, fixed_privilege)

                    # Priv filter, all measurements
                    gens = [g.pop() for g in sensor_generators[:fixed_privilege]]
                    priv_filters_all_ms[Y][Z] = PrivFusionFilter(n, m, F, Q, H, R, init_state, init_cov, Z_mat, Y_mat, gens, num_sensors)

            # Run simulation
            for _ in range(SIM_STEPS):
                
                # Update ground truth
                gt = ground_truth.update()
                sim.gt.append(gt)

                # Generate noise
                std_normals = np.block([g[0].next_n_as_std_gaussian(m) for g in sensor_generators])

                # For each of the parameter combinations compute estimates accordingly
                for Y in Ys:
                    for Z in Zs:
                        # Variables names of ease of reading (and likeness to other plot)
                        sensor_correlated_covariance = sensor_correlated_covariances[Y][Z]
                        unpriv_filter = unpriv_filters[Y][Z]

                        # Correlate noise
                        correlated_noises = np.linalg.cholesky(sensor_correlated_covariance)@std_normals

                        # Make all measurements and add pseudorandom noises
                        zs = []
                        for sen in range(num_sensors):
                            true_z = sensors[sen].measure(gt)
                            z = true_z + correlated_noises[sen*m:sen*m+m]
                            sim.zs[Y][Z][sen].append(z)
                            zs.append(z)

                        # Unpriv filter estimate
                        unpriv_filter.predict()
                        res = unpriv_filter.update(np.block(zs))
                        sim.unpriv_filters_results[Y][Z].append(res)

                        # Priv filter with denoise measurements only estimate
                        priv_filters_j_ms[Y][Z].predict()
                        res_j = priv_filters_j_ms[Y][Z].update(np.block(zs[:fixed_privilege]))
                        sim.priv_filters_j_ms_results[Y][Z].append(res_j)

                        # Priv filter with all measurements estimate
                        priv_filters_all_ms[Y][Z].predict()
                        res_all = priv_filters_all_ms[Y][Z].update(np.block(zs))
                        sim.priv_filters_all_ms_results[Y][Z].append(res_all)
        
            # Compute errors of the filters
            sim.compute_errors()

        # Average simulations
        avg_sim_data = AvgParameterSimData(sims)

        # Save sims
        all_sims['params'] = avg_sim_data

    """

     ######   ######     ###    ##    ## 
    ##    ## ##    ##   ## ##   ###   ## 
    ##       ##        ##   ##  ####  ## 
     ######  ##       ##     ## ## ## ## 
          ## ##       ######### ##  #### 
    ##    ## ##    ## ##     ## ##   ### 
     ######   ######  ##     ## ##    ## 
    
    """
    if DO_SCAN_SIM:
        print("\nMaking Parameter Scan Plot ...\n")
        
        # # Progress printing
        # if s % PROGRESS_PRINTS == 0:
        #     print("Running Simulation %d ..." % s)
        
        # Varying pseudorandom correlated and uncorrelated covariances

        Y_fixed = 5
        Z_fixed = 5
        Ys = np.arange(0.25,10.25,0.25)
        Zs = np.arange(0.25,10.25,0.25)

        # Only one privilege (number of keys) considered in this plot

        fixed_privileges = [1,2]

        # Sim data storage
        sim = ParameterScanSimData(0, num_sensors, Y_fixed, Z_fixed, Ys, Zs, fixed_privileges)

        # Synced cryptographically random number generators. Makes exactly the amount of each required by the simulation.
        l = len(Ys)+len(Zs)
        sensor_generators = [SharedKeyStreamFactory.make_shared_key_streams(1+4*l-2*l*(s*(s-(s%2))//3)) for s in range(num_sensors)]

        # Creating simulation objects (ground truth, sensors and filters)
        ground_truth = GroundTruth(F, Q, gt_init_state)
        sensors = []
        for _ in range(num_sensors):
            sensors.append(SensorPure(n, m, H, R))

        # As correlation parameters are being changed, store all filters and correlation matrices in 2-D dictionaries
        sensor_correlated_covariances = {}
        unpriv_filters = {}
        priv_filters_j_ms = {}
        priv_filters_all_ms = {}
        for priv in fixed_privileges:
            sensor_correlated_covariances[priv] = dict((("Z_fixed", {}), ("Y_fixed", {})))
            unpriv_filters[priv] = dict((("Z_fixed", {}), ("Y_fixed", {})))
            priv_filters_j_ms[priv] = dict((("Z_fixed", {}), ("Y_fixed", {})))
            priv_filters_all_ms[priv] = dict((("Z_fixed", {}), ("Y_fixed", {})))

            for Y in Ys:
                # Correlation matrix
                Y_mat = Y*np.eye(2)
                Z_mat = Z_fixed*np.eye(2)
                sensor_correlated_covariances[priv]["Z_fixed"][Y] = np.block([[Z_mat+Y_mat if r==c else Z_mat for c in range(num_sensors)] for r in range(num_sensors)])

                # Unpriv filter
                unpriv_filters[priv]["Z_fixed"][Y] = PrivFusionFilter(n, m, F, Q, H, R, init_state, init_cov, Z_mat, Y_mat, [], num_sensors)

                # Priv filter, denoisable measurements only
                gens = [g.pop() for g in sensor_generators[:priv]]
                priv_filters_j_ms[priv]["Z_fixed"][Y] = PrivFusionFilter(n, m, F, Q, H, R, init_state, init_cov, Z_mat, Y_mat, gens, priv)

                # Priv filter, all measurements
                gens = [g.pop() for g in sensor_generators[:priv]]
                priv_filters_all_ms[priv]["Z_fixed"][Y] = PrivFusionFilter(n, m, F, Q, H, R, init_state, init_cov, Z_mat, Y_mat, gens, num_sensors)
            
            for Z in Zs:
                # Correlation matrix
                Y_mat = Y_fixed*np.eye(2)
                Z_mat = Z*np.eye(2)
                sensor_correlated_covariances[priv]["Y_fixed"][Z] = np.block([[Z_mat+Y_mat if r==c else Z_mat for c in range(num_sensors)] for r in range(num_sensors)])

                # Unpriv filter
                unpriv_filters[priv]["Y_fixed"][Z] = PrivFusionFilter(n, m, F, Q, H, R, init_state, init_cov, Z_mat, Y_mat, [], num_sensors)

                # Priv filter, denoisable measurements only
                gens = [g.pop() for g in sensor_generators[:priv]]
                priv_filters_j_ms[priv]["Y_fixed"][Z] = PrivFusionFilter(n, m, F, Q, H, R, init_state, init_cov, Z_mat, Y_mat, gens, priv)

                # Priv filter, all measurements
                gens = [g.pop() for g in sensor_generators[:priv]]
                priv_filters_all_ms[priv]["Y_fixed"][Z] = PrivFusionFilter(n, m, F, Q, H, R, init_state, init_cov, Z_mat, Y_mat, gens, num_sensors)



        # Run simulation
        for _ in range(SIM_STEPS):
            
            # Update ground truth
            gt = ground_truth.update()
            sim.gt.append(gt)

            # Generate noise
            std_normals = np.block([g[0].next_n_as_std_gaussian(m) for g in sensor_generators])

            # For each of the parameter combinations compute estimates accordingly
            for priv in fixed_privileges:
                for Y in Ys:
                    # Variables names of ease of reading (and likeness to other plot)
                    sensor_correlated_covariance = sensor_correlated_covariances[priv]["Z_fixed"][Y]
                    unpriv_filter = unpriv_filters[priv]["Z_fixed"][Y]

                    # Correlate noise
                    correlated_noises = np.linalg.cholesky(sensor_correlated_covariance)@std_normals

                    # Make all measurements and add pseudorandom noises
                    zs = []
                    for sen in range(num_sensors):
                        true_z = sensors[sen].measure(gt)
                        z = true_z + correlated_noises[sen*m:sen*m+m]
                        sim.zs[priv]["Z_fixed"][Y][sen].append(z)
                        zs.append(z)

                    # Unpriv filter estimate
                    unpriv_filter.predict()
                    res = unpriv_filter.update(np.block(zs))
                    sim.unpriv_filters_results[priv]["Z_fixed"][Y].append(res)

                    # Priv filter with denoise measurements only estimate
                    priv_filters_j_ms[priv]["Z_fixed"][Y].predict()
                    res_j = priv_filters_j_ms[priv]["Z_fixed"][Y].update(np.block(zs[:priv]))
                    sim.priv_filters_j_ms_results[priv]["Z_fixed"][Y].append(res_j)

                    # Priv filter with all measurements estimate
                    priv_filters_all_ms[priv]["Z_fixed"][Y].predict()
                    res_all = priv_filters_all_ms[priv]["Z_fixed"][Y].update(np.block(zs))
                    sim.priv_filters_all_ms_results[priv]["Z_fixed"][Y].append(res_all)
                
                for Z in Zs:
                    # Variables names of ease of reading (and likeness to other plot)
                    sensor_correlated_covariance = sensor_correlated_covariances[priv]["Y_fixed"][Z]
                    unpriv_filter = unpriv_filters[priv]["Y_fixed"][Z]

                    # Correlate noise
                    correlated_noises = np.linalg.cholesky(sensor_correlated_covariance)@std_normals

                    # Make all measurements and add pseudorandom noises
                    zs = []
                    for sen in range(num_sensors):
                        true_z = sensors[sen].measure(gt)
                        z = true_z + correlated_noises[sen*m:sen*m+m]
                        sim.zs[priv]["Y_fixed"][Z][sen].append(z)
                        zs.append(z)

                    # Unpriv filter estimate
                    unpriv_filter.predict()
                    res = unpriv_filter.update(np.block(zs))
                    sim.unpriv_filters_results[priv]["Y_fixed"][Z].append(res)

                    # Priv filter with denoise measurements only estimate
                    priv_filters_j_ms[priv]["Y_fixed"][Z].predict()
                    res_j = priv_filters_j_ms[priv]["Y_fixed"][Z].update(np.block(zs[:priv]))
                    sim.priv_filters_j_ms_results[priv]["Y_fixed"][Z].append(res_j)

                    # Priv filter with all measurements estimate
                    priv_filters_all_ms[priv]["Y_fixed"][Z].predict()
                    res_all = priv_filters_all_ms[priv]["Y_fixed"][Z].update(np.block(zs))
                    sim.priv_filters_all_ms_results[priv]["Y_fixed"][Z].append(res_all)
    
        # Compute errors of the filters
        sim.compute_steady_state_traces()

        # Save sims
        all_sims['scan'] = sim
    
    # Store to disk
    pickle.dump(all_sims, open("code/priv_fusion_sims.p", "wb"))

    return

"""
 
 ########  ##        #######  ########  ######  
 ##     ## ##       ##     ##    ##    ##    ## 
 ##     ## ##       ##     ##    ##    ##       
 ########  ##       ##     ##    ##     ######  
 ##        ##       ##     ##    ##          ## 
 ##        ##       ##     ##    ##    ##    ## 
 ##        ########  #######     ##     ######  
 
"""
def plot_privilege_differences():
    all_sims = pickle.load(open("code/priv_fusion_sims.p", "rb"))
    avg_sim_data = all_sims['privs']

    # Suggested defaults: wspace=0.2, hspace=0.2, top=0.9, bottom=0.1, left=0.125, right=0.9
    fig, axes = plt.subplots(2, 2, figsize=(0.7*5.78853, 0.7*5.78853), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.2, hspace=0.22, top=0.8, bottom=0.15, left=0.15, right=0.85)

    # Loop and make the plots
    unpriv_plots = []
    priv_denoised_plots = []
    priv_all_plots = []
    for i,ax in enumerate(axes.flat):
        ax.grid(True, c='lightgray')

        ax.set_title(r'$\pi=%d$' % (i+1))

        # Priv all at each privilege
        pa, = ax.plot([x for x in range(avg_sim_data.sim_len)], [e for e in avg_sim_data.priv_filters_all_ms_errors_avg[i]], linestyle='-', color='tab:blue')
        # Priv all trace (to check the average MSE above is correct)
        #pa, = ax.plot([x for x in range(avg_sim_data.sim_len)], [np.trace(e[1]) for e in avg_sim_data.last_sim.priv_filters_all_ms_results[i]], linestyle='-')
        
        # Priv only denoised at each privilege
        pd, = ax.plot([x for x in range(avg_sim_data.sim_len)], [e for e in avg_sim_data.priv_filters_j_ms_errors_avg[i]], linestyle='-', color='tab:green')
        # Priv only denoised trace (to check the average MSE above is correct)
        #pd, = ax.plot([x for x in range(avg_sim_data.sim_len)], [np.trace(e[1]) for e in avg_sim_data.last_sim.priv_filters_j_ms_results[i]], linestyle='-')

        # Unpriv in each plot
        u, = ax.plot([x for x in range(avg_sim_data.sim_len)], [e for e in avg_sim_data.unpriv_filter_errors_avg], linestyle='-', color='tab:red')
        # Unpriv trace (to check the average MSE above is correct)
        #u, = ax.plot([x for x in range(avg_sim_data.sim_len)], [np.trace(e[1]) for e in avg_sim_data.last_sim.unpriv_filter_results], linestyle='-')

        # Differences controlled by bounds
        pllb = ax.fill_between([x for x in range(avg_sim_data.sim_len)], [e for e in avg_sim_data.priv_filters_j_ms_errors_avg[i]], [e for e in avg_sim_data.unpriv_filter_errors_avg], color='tab:orange', alpha=0.15)
        pgub = ax.fill_between([x for x in range(avg_sim_data.sim_len)], [e for e in avg_sim_data.priv_filters_j_ms_errors_avg[i]], [e for e in avg_sim_data.priv_filters_all_ms_errors_avg[i]], color='tab:purple', alpha=0.15)

        unpriv_plots.append(u)
        priv_denoised_plots.append(pd)
        priv_all_plots.append(pa)

        #ax.set_yticks([0, 1, 2])

    # Legend
    fig.legend((priv_all_plots[0],
                priv_denoised_plots[0], 
                unpriv_plots[0], 
                pllb,
                pgub), 
               (r'$\mathsf{e}^{[\pi,4]}$',
                r'$\mathsf{e}^{[\pi,\pi]}$', 
                r'$\mathsf{e}^{[0,4]}$',
                r'Error diff. bound by $\mathsf{tr}(\mat{D}_{\mathsf{PLLB}})$',
                r'Error diff. bound by $\mathsf{tr}(\mat{D}_{\mathsf{PGUB}})$',), loc='upper center', ncol=2)

    # Shared axis labels
    fig.supxlabel(r'Simulation Timestep')   
    fig.supylabel(r'Mean Square Error (MSE)')

    # Hide relevant axis ticks
    for a in [axes[0][0], axes[0][1]]:
        a.tick_params(bottom=False)
    for a in [axes[0][1], axes[1][1]]:
        a.tick_params(left=False)

    # Save or show figure
    if SAVE_FIGS:
        plt.savefig('figures/priv_estimation_fus_mse_privs.pdf')
    else:
        plt.show()

    return


def plot_parameter_differences():
    all_sims = pickle.load(open("code/priv_fusion_sims.p", "rb"))
    avg_sim_data = all_sims['params']

    # Suggested defaults: wspace=0.2, hspace=0.2, top=0.9, bottom=0.1, left=0.125, right=0.9
    fig, axes = plt.subplots(2, 2, figsize=(0.7*5.78853, 0.7*5.78853), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.2, hspace=0.22, top=0.8, bottom=0.15, left=0.15, right=0.85)

    # Loop and make the plots
    unpriv_plots = []
    priv_denoised_plots = []
    priv_all_plots = []
    ind = 0
    for Y in avg_sim_data.Ys:
        for Z in avg_sim_data.Zs:
            ax = axes.flat[ind]
            ax.grid(True, c='lightgray')

            ax.set_title(r'$V=%.0lf$, $W=%.0lf$' % (Z, Y))

            # Priv all at each privilege
            pa, = ax.plot([x for x in range(avg_sim_data.sim_len)], [e for e in avg_sim_data.priv_filters_all_ms_errors_avg[Y][Z]], linestyle='-', color='tab:blue')
            # Priv all trace (to check the average MSE above is correct)
            #pa, = ax.plot([x for x in range(avg_sim_data.sim_len)], [np.trace(e[1]) for e in avg_sim_data.last_sim.priv_filters_all_ms_results[Y][Z]], linestyle='-')
            
            # Priv only denoised at each privilege
            pd, = ax.plot([x for x in range(avg_sim_data.sim_len)], [e for e in avg_sim_data.priv_filters_j_ms_errors_avg[Y][Z]], linestyle='-', color='tab:green')
            # Priv only denoised trace (to check the average MSE above is correct)
            #pd, = ax.plot([x for x in range(avg_sim_data.sim_len)], [np.trace(e[1]) for e in avg_sim_data.last_sim.priv_filters_j_ms_results[Y][Z]], linestyle='-')

            # Unpriv in each plot
            u, = ax.plot([x for x in range(avg_sim_data.sim_len)], [e for e in avg_sim_data.unpriv_filters_errors_avg[Y][Z]], linestyle='-', color='tab:red')
            # Unpriv trace (to check the average MSE above is correct)
            #u, = ax.plot([x for x in range(avg_sim_data.sim_len)], [np.trace(e[1]) for e in avg_sim_data.last_sim.unpriv_filters_results[Y][Z]], linestyle='-')

            # Differences controlled by bounds
            pllb = ax.fill_between([x for x in range(avg_sim_data.sim_len)], [e for e in avg_sim_data.priv_filters_j_ms_errors_avg[Y][Z]], [e for e in avg_sim_data.unpriv_filters_errors_avg[Y][Z]], color='tab:orange', alpha=0.15)
            pgub = ax.fill_between([x for x in range(avg_sim_data.sim_len)], [e for e in avg_sim_data.priv_filters_j_ms_errors_avg[Y][Z]], [e for e in avg_sim_data.priv_filters_all_ms_errors_avg[Y][Z]], color='tab:purple', alpha=0.15)

            unpriv_plots.append(u)
            priv_denoised_plots.append(pd)
            priv_all_plots.append(pa)

            #ax.set_yticks([0, 2, 4])

            ind+=1

    # Legend
    fig.legend((priv_all_plots[0],
                priv_denoised_plots[0], 
                unpriv_plots[0], 
                pllb,
                pgub), 
               (r'$\textsf{e}^{[2,4]}$',
                r'$\textsf{e}^{[2,2]}$',
                r'$\textsf{e}^{[0,4]}$',
                r'Error diff. bound by $\mathsf{tr}(\mat{D}_{\mathsf{PLLB}})$',
                r'Error diff. bound by $\mathsf{tr}(\mat{D}_{\mathsf{PGUB}})$',), loc='upper center', ncol=2)

    # Shared axis labels
    fig.supxlabel(r'Simulation Timestep')
    fig.supylabel(r'Mean Square Error (MSE)')

    # Hide relevant axis ticks
    for a in [axes[0][0], axes[0][1]]:
        a.tick_params(bottom=False)
    for a in [axes[0][1], axes[1][1]]:
        a.tick_params(left=False)

    # Save or show figure
    if SAVE_FIGS:
        plt.savefig('figures/priv_estimation_fus_mse_params.pdf')
    else:
        plt.show()

    return


def plot_parameter_scan():
    all_sims = pickle.load(open("code/priv_fusion_sims.p", "rb"))
    sim_data = all_sims['scan']

    # Suggested defaults: wspace=0.2, hspace=0.2, top=0.9, bottom=0.1, left=0.125, right=0.9
    fig, axes = plt.subplots(2, 2, figsize=(0.7*5.78853, 0.7*5.78853), sharex='col', sharey=True)
    plt.subplots_adjust(wspace=0.2, hspace=0.22, top=0.85, bottom=0.15, left=0.15, right=0.85)

    # Loop and make the plots
    unpriv_plots = []
    priv_denoised_plots = []
    priv_all_plots = []
    ind = 0
    for priv in sim_data.privileges:
        for fixed in ["Z_fixed", "Y_fixed"]:

            ax = axes.flat[ind]
            ax.grid(True, c='lightgray')
            ax.set_title(r'$\pi=%d$, $%s=%d$' % (priv, r'W' if ind%2==1 else r'V', sim_data.Y_fixed if ind%2==1 else sim_data.Z_fixed))

            if fixed == 'Y_fixed':
                x = sim_data.Ys
            else:
                x = sim_data.Zs

            # Priv only denoised at each privilege
            pd, = ax.plot(x, [b-a for a,b in zip([t for t in sim_data.priv_filters_j_ms_traces[priv][fixed]],[t for t in sim_data.priv_filters_all_ms_traces[priv][fixed]])], linestyle='-', color='tab:blue')
            
            # Unpriv in each plot
            u, = ax.plot(x, [a-b for a,b in zip([t for t in sim_data.unpriv_filters_traces[priv][fixed]],[t for t in sim_data.priv_filters_j_ms_traces[priv][fixed]])], linestyle='-', color='tab:green')
            
            l = ax.hlines(0, x[0], x[-1], colors=['tab:orange'], linestyle=':')

            # Priv all at each privilege
            #pa, = ax.plot(x, [t for t in sim_data.priv_filters_all_ms_traces[priv][fixed]], linestyle='-.')

            unpriv_plots.append(u)
            priv_denoised_plots.append(pd)
            #priv_all_plots.append(pa)

            # Column x axis labels
            if priv == sim_data.privileges[-1]:
                #ax.set_xlabel(r'$\Sigma_%s$' % ('V' if ind%2==1 else 'W'), size='large')
                ax.set_xlabel(r'$%s$' % ('V' if ind%2==1 else 'W'))

            ind+=1

    # Legend
    fig.legend((priv_denoised_plots[0],
                unpriv_plots[0], 
                l),
               (r'PLLB',
                r'PGUB',
                r'$0$'), loc='upper center', ncol=3)

    # Shared axis labels
    fig.supylabel(r'Steady-State Trace $\Bigl(\mathsf{tr}(\lim_{k \to \infty}\mathbf{D}_k)\Bigr)$')

    # Hide relevant axis ticks
    for a in [axes[0][0], axes[0][1]]:
        a.tick_params(bottom=False)
    for a in [axes[0][1], axes[1][1]]:
        a.tick_params(left=False)

    # Save or show figure
    if SAVE_FIGS:
        plt.savefig('figures/priv_estimation_fus_trace_params.pdf')
    else:
        plt.show()

    return




#run_simulations()
#plot_privilege_differences()
plot_parameter_differences()
plot_parameter_scan()
