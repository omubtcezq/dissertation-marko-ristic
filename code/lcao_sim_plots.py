import numpy as np
import pickle
import matplotlib.pyplot as plt
import plotting as pltng

SAVE_FIGS = True
pltng.init_matplotlib_params(SAVE_FIGS, True)

LAYOUT_TRACK_FILEPATH = 'C:/Users/ristic/Documents/GIT_REPOS/PrivateEIFLocalisationCode/input/debug_track_001.txt'
ESTIMATION_ERRORS_FILEPATH_BASE = 'C:/Users/ristic/Documents/GIT_REPOS/PrivateEIFLocalisationCode/output_evaluation/layout_%s_nav_mean_errors.txt'
ESTIMATION_ERRORS_CONTROL_FILEPATH_BASE = 'C:/Users/ristic/Documents/GIT_REPOS/PrivateEIFLocalisationCode/output_evaluation/eif_layout_%s_nav_mean_errors.txt'
TIMING_FILEPATH_BASE = 'C:/Users/ristic/Documents/GIT_REPOS/PrivateEIFLocalisationCode/output/timing_%d_%d_nav_times.txt'
SENSOR_LOCATIONS = [[np.array([5.0, 5.0]), # Normal
                     np.array([40.0, 5.0]), 
                     np.array([5.0, 40.0]), 
                     np.array([40.0, 40.0])],
                    [np.array([-30.0, -30.0]), # Big
                     np.array([75.0, -30.0]), 
                     np.array([-30.0, 75.0]), 
                     np.array([75.0, 75.0])],
                    [np.array([-65.0, -65.0]), # Very big 
                     np.array([110.0, -65.0]), 
                     np.array([-65.0, 110.0]), 
                     np.array([110.0, 110.0])],
                    [np.array([-100.0, -100.0]), # Huge
                     np.array([145.0, -100.0]), 
                     np.array([-100.0, 145.0]), 
                     np.array([145.0, 145.0])]]
LAYOUT_TAGS = ['normal', 'big', 'verybig', 'huge']
LAYOUT_NAMES = ['Normal', 'Big', 'Quite Big', 'Very Big']
ESTIMATION_SIM_TIMESTEPS = 50
TIMING_SENSOR_COUNTS = [2,3,4,5]
TIMING_BIT_LENGTHS = [512, 1024, 1536, 2048, 2560]


"""
 
  ######  #### ##     ##  ######  
 ##    ##  ##  ###   ### ##    ## 
 ##        ##  #### #### ##       
  ######   ##  ## ### ##  ######  
       ##  ##  ##     ##       ## 
 ##    ##  ##  ##     ## ##    ## 
  ######  #### ##     ##  ######  
 
"""
def create_python_sim_data():
    sims = {}

    # Layouts
    with open(LAYOUT_TRACK_FILEPATH) as track_f:
        ground_truth = []
        timesteps = int(track_f.readline())
        dimenstions = int(track_f.readline())
        init_state = np.array([float(x) for x in track_f.readline().split()])
        init_cov = np.array([[float(x) for x in track_f.readline().split()] for _ in range(dimenstions)])
        for _ in range(timesteps):
            ground_truth.append(np.array([float(x) for x in track_f.readline().split()]))
    sims['layout'] = {'gt': ground_truth, 'init_state':init_state, 'init_cov':init_cov}

    # Estimation
    sims['estimation'] = {}
    for layout in LAYOUT_TAGS:
        with open(ESTIMATION_ERRORS_FILEPATH_BASE % layout) as distance_f:
            mean_enc_errors = [float(x.strip()) if x.strip() != 'Failed' else -1 for x in distance_f.read().split()]
        with open(ESTIMATION_ERRORS_CONTROL_FILEPATH_BASE % layout) as eif_distance_f:
            mean_eif_errors = [float(x.strip()) if x.strip() != 'Failed' else -1 for x in eif_distance_f.read().split()]
        sims['estimation'][layout] = {'mean_enc_errors': mean_enc_errors, 'mean_eif_errors': mean_eif_errors}

    # Timing
    TIMING_BIT_LENGTHS.sort(reverse=True)
    sims['timing'] = {}
    for s in TIMING_SENSOR_COUNTS:
        sims['timing'][s] = []
        for b in TIMING_BIT_LENGTHS:
            with open(TIMING_FILEPATH_BASE % (b, s)) as timing_f:
                time_vals = [float(x.strip()) for x in timing_f.read().split() if x.strip() != 'Failed']
                if len(time_vals) == 0:
                    mean = -1
                else:
                    mean = np.mean(time_vals)
                sims['timing'][s].append(mean)

        assert(len(sims['timing'][s]) == len(TIMING_BIT_LENGTHS))

    pickle.dump(sims, open("code/nonlin_fusion_sims.p", "wb"))
    return

"""
 
 ##          ###    ##    ##  #######  ##     ## ########  ######  
 ##         ## ##    ##  ##  ##     ## ##     ##    ##    ##    ## 
 ##        ##   ##    ####   ##     ## ##     ##    ##    ##       
 ##       ##     ##    ##    ##     ## ##     ##    ##     ######  
 ##       #########    ##    ##     ## ##     ##    ##          ## 
 ##       ##     ##    ##    ##     ## ##     ##    ##    ##    ## 
 ######## ##     ##    ##     #######   #######     ##     ######  
 
"""
def layouts_plot():
    # Read store sim data
    sims = pickle.load(open("code/nonlin_fusion_sims.p", "rb"))
    ground_truth = sims['layout']['gt']
    init_state = sims['layout']['init_state']
    init_cov = sims['layout']['init_cov']

    # Make subplots. 70% text width and default matplotlib aspect ratio (0.7*5.78853, 0.75*0.7*5.78853),
    fig, axs = plt.subplots(2,2, figsize=(0.7*5.78853, 0.75*0.7*5.78853), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.05, hspace=0.3, top=0.82, bottom=0.15, left=0.15, right=0.85)

    plots = []
    scatters = []
    for i,ax in enumerate(axs.flat):
        ax.set_title(LAYOUT_NAMES[i])
        ax.set_aspect(aspect='equal')
        # Plot sensor positions and groundtruth
        s = ax.scatter([x[0] for x in SENSOR_LOCATIONS[i]], [x[1] for x in SENSOR_LOCATIONS[i]], marker='.', color='tab:orange')
        p, = ax.plot([x[0] for x in ground_truth], [x[2] for x in ground_truth], color='tab:blue')
        i_s = ax.scatter([init_state[0]], [init_state[2]], marker='.', color='tab:green')
        ax.add_artist(pltng.get_cov_ellipse(np.array([[init_cov[0][0], init_cov[0][2]],
                                                      [init_cov[2][0], init_cov[2][2]]]), 
                                            np.array([init_state[0],init_state[2]]), 
                                            2, fill=False, linestyle='-', edgecolor='tab:green', zorder=1))
        # Save the plots for the legend
        scatters.append(s)
        plots.append(p)

    # Legend
    fig.legend((plots[0], i_s, scatters[0]), (r'Ground Truth', r'Initial Estimate', r'Sensors'), loc='upper center', ncol=3)
    # Shared axis labels
    fig.supxlabel(r'Location $x$')
    fig.supylabel(r'Location $y$')
    # Hide ticks from intermediate axes
    for a in [axs[0][0], axs[0][1]]:
        a.tick_params(bottom=False)
    for a in [axs[0][1], axs[1][1]]:
        a.tick_params(left=False)

    if SAVE_FIGS:
        plt.savefig('figures/nonlin_fusion_simulation_layouts.pdf')
    else:
        plt.show()
    plt.close()
    return

"""
 
 ########  ######  ######## #### ##     ##    ###    ######## ####  #######  ##    ## 
 ##       ##    ##    ##     ##  ###   ###   ## ##      ##     ##  ##     ## ###   ## 
 ##       ##          ##     ##  #### ####  ##   ##     ##     ##  ##     ## ####  ## 
 ######    ######     ##     ##  ## ### ## ##     ##    ##     ##  ##     ## ## ## ## 
 ##             ##    ##     ##  ##     ## #########    ##     ##  ##     ## ##  #### 
 ##       ##    ##    ##     ##  ##     ## ##     ##    ##     ##  ##     ## ##   ### 
 ########  ######     ##    #### ##     ## ##     ##    ##    ####  #######  ##    ## 
 
"""
def estimation_plot():
    # Read store sim data
    sims = pickle.load(open("code/nonlin_fusion_sims.p", "rb"))

    # Width fixed for template, the rest is eye-balled
    fig, axs = plt.subplots(4,1, figsize=(0.7*5.78853, 1.5*0.7*5.78853), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.5, top=0.9)
    
    enc_plot_handles = []
    eif_plot_handles = []
    for i in range(len(LAYOUT_NAMES)):
        ax = axs.flat[i]
        ax.set_title(LAYOUT_NAMES[i])
        ax.grid()
        # Hide bottom tick of all but bottom plot
        if i < len(LAYOUT_NAMES)-1:
            ax.tick_params(bottom=False)
        # Plot error of our method
        mean_enc_errors = sims['estimation'][LAYOUT_TAGS[i]]['mean_enc_errors']
        # Skip first point (initial point is good estimate with bad covariance)
        ph_enc, = ax.plot([x for x in list(range(ESTIMATION_SIM_TIMESTEPS))][1:], mean_enc_errors[1:], color='tab:blue')
        enc_plot_handles.append(ph_enc)
        # Plot error of normal eif
        mean_eif_errors = sims['estimation'][LAYOUT_TAGS[i]]['mean_eif_errors']
        ph_eif, = ax.plot([x for x in list(range(ESTIMATION_SIM_TIMESTEPS))][1:], mean_eif_errors[1:], color='tab:orange', linestyle=':')
        eif_plot_handles.append(ph_eif)

    # Shared axis labels
    fig.supxlabel(r'Simulation Timestep')
    fig.supylabel(r'Mean Square Error (MSE)')

    # Legend only uses lines from first plot (all are the same colours)
    fig.legend((enc_plot_handles[0], eif_plot_handles[0]), (r'Confidential Filter', r'Unmodified EIF'), loc='upper center', ncol=2)

    if SAVE_FIGS:
        plt.savefig('figures/nonlin_fusion_simulation_layout_errors.pdf')
    else:
        plt.show()
    plt.close()
    return

"""
 
 ######## #### ##     ## #### ##    ##  ######   
    ##     ##  ###   ###  ##  ###   ## ##    ##  
    ##     ##  #### ####  ##  ####  ## ##        
    ##     ##  ## ### ##  ##  ## ## ## ##   #### 
    ##     ##  ##     ##  ##  ##  #### ##    ##  
    ##     ##  ##     ##  ##  ##   ### ##    ##  
    ##    #### ##     ## #### ##    ##  ######   
 
"""
def timing_plot():
    TIMING_BIT_LENGTHS.sort(reverse=True)
    # Read store sim data
    sims = pickle.load(open("code/nonlin_fusion_sims.p", "rb"))
    sensor_counts = sims['timing']

    # Fixed width for template, the rest if eye-balled
    fig = plt.figure()
    fig.set_size_inches(w=0.7*5.78853, h=0.75*0.7*5.78853)
    ax = fig.add_subplot(111)
    ax.grid()
    plt.subplots_adjust(top=0.75)
    #ax.grid(linestyle='dashed')
    #ax.set_axisbelow(True)

    plot_handles = []
    for b in range(len(TIMING_BIT_LENGTHS)):
        ph, = ax.plot(TIMING_SENSOR_COUNTS, [sensor_counts[i][b] for i in TIMING_SENSOR_COUNTS], marker='.', label=r'%d' % TIMING_BIT_LENGTHS[b])
        plot_handles.append(ph)

    # Set the ticks to be all of the sensor amounts tested
    ax.set_xticks(TIMING_SENSOR_COUNTS)
    ax.set_yticks([0,50,100,150])

    # Used gloabl axis labels to match the other figures
    ax.set_xlabel(r'Number of Sensors')
    ax.set_ylabel(r'Runtime ($s$)')

    # Legend
    fig.legend(handles=plot_handles, title='Key Length (bits)', loc='upper center', ncol=3)

    if SAVE_FIGS:
        plt.savefig('figures/nonlin_fusion_simulation_timing.pdf')
    else:
        plt.show()
    plt.close()
    return






#create_python_sim_data()
layouts_plot()
estimation_plot()
timing_plot()