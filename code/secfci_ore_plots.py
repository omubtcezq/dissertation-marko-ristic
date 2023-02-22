import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri

import plotting as pltng

SAVE_FIGS = True
pltng.init_matplotlib_params(SAVE_FIGS, True)



"""
 
  #######   ######  ######## ##    ## 
 ##     ## ##    ## ##       ###   ## 
        ## ##       ##       ####  ## 
  #######   ######  ######   ## ## ## 
 ##              ## ##       ##  #### 
 ##        ##    ## ##       ##   ### 
 #########  ######  ######## ##    ## 
 
"""
def two_sen():
    fig = plt.figure()
    # 70% text width and default matplotlib aspect ratio
    fig.set_size_inches(w=0.7*5.78853, h=0.75*0.7*5.78853)
    ax = fig.add_subplot(111)
    # ax.grid(True)

    g = 0.1
    gs = np.arange(0, 1+g, g)

    trA = 7.6
    trB = 2.4

    A = gs * trA
    B = (1-gs) * trB

    g_cmps = gs*trA > (1-gs)*trB
    l = r = 0
    for i,b in enumerate(g_cmps):
        if not b:
            l = r = i
        else:
            r = i
            break
    l = gs[l]
    r = gs[r]

    ax.plot(gs, A, marker='.', c='tab:blue', label=r'$\omega_1\tr(\mat{P}_1)$', zorder=3)
    ax.plot(gs, B, marker='.', c='tab:orange', label=r'$(1-\omega_1)\tr(\mat{P}_2)$', zorder=3)

    ax.scatter([l,r],[0, 0], marker='.', c='tab:grey', zorder=2, label=r'Solution Bounds')
    ax.plot([l, l],[0, (1-l)*trB], linestyle='--', c='tab:grey', zorder=2)
    ax.plot([r, r],[0, r*trA], linestyle='--', c='tab:grey', zorder=2)
    ax.scatter([0.5*(l+r)],[0], marker='.', c='tab:green', zorder=1, label=r'Approximate Solution')

    plt.xlabel(r'$\omega_1$')

    plt.legend()
    ax.set_yticks([trA, trB])
    ax.set_yticklabels([r'$\tr(\mat{P}_1)$', r'$\tr(\mat{P}_2)$'])
    ax.set_xticks(list(np.arange(0,1.1,0.1))+[0.5*(l+r)])
    ax.set_xticklabels([str(x/10) if x in [0,5,10] else '' for x in range(11)]+[r'$\hat{\omega}_1$'])

    if SAVE_FIGS:
        plt.savefig('figures/cloud_fusion_secfci_2sen_intersect.pdf')
    else:
        plt.show()
    plt.close()
    return


"""
 
 ########      ######   #######  ##             ##   
 ##     ##    ##    ## ##     ## ##           ####   
 ##     ##    ##       ##     ## ##             ##   
 ########      ######  ##     ## ##             ##   
 ##                 ## ##     ## ##             ##   
 ##           ##    ## ##     ## ##             ##   
 ##            ######   #######  ########     ###### 
 
"""
def p_sol_1():
    fig = plt.figure()
    fig.set_size_inches(w=0.8*5.78853, h=0.75*0.5*5.78853)
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    fig.subplots_adjust(right=0.6, bottom=0.2, top=0.95)

    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)
    ax.set_xlabel(r'$\omega_1$')
    ax.set_ylabel(r'$\omega_2$')
    ax.set_zlabel(r'$\omega_3$')
    ax.view_init(elev=35, azim=2)
    ax.set_box_aspect((4,4,3), zoom=1)

    # Solution plane
    xy = np.array([[1,0],
                   [0,0],
                   [0,1]])
    z = np.array([0,1,0])
    trigs = [[0,1,2]]
    triangles = mtri.Triangulation(xy[:,0], xy[:,1], triangles=trigs)
    ax.plot_trisurf(triangles, z, color='tab:grey', alpha=0.4, zorder=1)


    # Partial solution 1
    sol1 = 0.24
    sol1Line, = ax.plot([sol1, 0],[1-sol1, 0],[0, 1], linestyle='--', c='tab:blue', zorder=5)
    ax.scatter([sol1, 0],[1-sol1, 0],[0, 1], c='tab:blue', marker='.', depthshade=False, zorder=10)
    ax.text(sol1+0.2, 1-sol1, 0, r'$\vec{\omega}_1^{(2)}$')
    ax.text(0, 0+0.1, 1-0.05, r'$\vec{\omega}_1^{(1)}$')

    solutionSurfaceFakeLine = mpl.lines.Line2D([0],[0], linestyle="none", c='tab:grey', marker = 'o')
    ax.legend([solutionSurfaceFakeLine, sol1Line], 
              [r'$\sum_{i=1}^n\omega_i=1$', r'Solution Subspace $\mathcal{S}_1(\vec{\gamma})$'],
               loc='center left',
               bbox_to_anchor=(1.15, 0.5))

    ax.xaxis.set_ticks([0,0.5,1])
    ax.yaxis.set_ticks([0,0.5,1])
    ax.zaxis.set_ticks([0,0.5,1])

    if SAVE_FIGS:
        plt.savefig('figures/cloud_fusion_secfci_partial_sol_1.pdf')
    else:
        plt.show()
    plt.close()


"""
 
 ########      ######   #######  ##           #######  
 ##     ##    ##    ## ##     ## ##          ##     ## 
 ##     ##    ##       ##     ## ##                 ## 
 ########      ######  ##     ## ##           #######  
 ##                 ## ##     ## ##          ##        
 ##           ##    ## ##     ## ##          ##        
 ##            ######   #######  ########    ######### 
 
"""
def p_sol_2():
    fig = plt.figure()
    fig.set_size_inches(w=0.8*5.78853, h=0.75*0.5*5.78853)
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    fig.subplots_adjust(right=0.6, bottom=0.2, top=0.95)

    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)
    ax.set_xlabel(r'$\omega_1$')
    ax.set_ylabel(r'$\omega_2$')
    ax.set_zlabel(r'$\omega_3$')
    ax.view_init(elev=35, azim=2)
    ax.set_box_aspect((4,4,3), zoom=1)

    # Solution plane
    xy = np.array([[1,0],
                   [0,0],
                   [0,1]])
    z = np.array([0,1,0])
    trigs = [[0,1,2]]
    triangles = mtri.Triangulation(xy[:,0], xy[:,1], triangles=trigs)
    ax.plot_trisurf(triangles, z, color='tab:grey', alpha=0.4, zorder=1)


    # Partial solution 2
    sol2 = 0.52
    sol2Line, = ax.plot([0, 1],[sol2, 0],[1-sol2, 0], linestyle='--', c='tab:orange', zorder=5)
    ax.scatter([0, 1],[sol2, 0],[1-sol2, 0], c='tab:orange', marker='.', depthshade=False, zorder=10)
    ax.text(0, sol2+0.07, 1-sol2+0.07, r'$\vec{\omega}_2^{(2)}$')
    ax.text(1, 0+0.1, 0, r'$\vec{\omega}_2^{(1)}$')

    solutionSurfaceFakeLine = mpl.lines.Line2D([0],[0], linestyle="none", c='tab:grey', marker = 'o')
    ax.legend([solutionSurfaceFakeLine, sol2Line], 
              [r'$\sum_{i=1}^n\omega_i=1$', r'Solution Subspace $\mathcal{S}_2(\vec{\gamma})$'],
               loc='center left',
               bbox_to_anchor=(1.15, 0.5))

    ax.xaxis.set_ticks([0,0.5,1])
    ax.yaxis.set_ticks([0,0.5,1])
    ax.zaxis.set_ticks([0,0.5,1])

    if SAVE_FIGS:
        plt.savefig('figures/cloud_fusion_secfci_partial_sol_2.pdf')
    else:
        plt.show()
    plt.close()



"""
 
 ########      ######   #######  ##          #### ##    ## ######## ######## ########  
 ##     ##    ##    ## ##     ## ##           ##  ###   ##    ##    ##       ##     ## 
 ##     ##    ##       ##     ## ##           ##  ####  ##    ##    ##       ##     ## 
 ########      ######  ##     ## ##           ##  ## ## ##    ##    ######   ########  
 ##                 ## ##     ## ##           ##  ##  ####    ##    ##       ##   ##   
 ##           ##    ## ##     ## ##           ##  ##   ###    ##    ##       ##    ##  
 ##            ######   #######  ########    #### ##    ##    ##    ######## ##     ## 
 
"""
def p_sol_intersection():
    fig = plt.figure()
    fig.set_size_inches(w=0.8*5.78853, h=0.75*0.5*5.78853)
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    fig.subplots_adjust(right=0.6, bottom=0.2, top=0.95)

    
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.zaxis.set_rotate_label(False)
    ax.set_xlabel(r'$\omega_1$')
    ax.set_ylabel(r'$\omega_2$')
    ax.set_zlabel(r'$\omega_3$')
    ax.view_init(elev=35, azim=2)
    ax.set_box_aspect((4,4,3), zoom=1)

    # Solution plane
    xy = np.array([[1,0],
                   [0,0],
                   [0,1]])
    z = np.array([0,1,0])
    trigs = [[0,1,2]]
    triangles = mtri.Triangulation(xy[:,0], xy[:,1], triangles=trigs)
    ax.plot_trisurf(triangles, z, color='tab:grey', alpha=0.4, zorder=1)

    # Partial solution 1
    sol1 = 0.24
    sol1Line, = ax.plot([sol1, 0],[1-sol1, 0],[0, 1], linestyle='--', c='tab:blue', zorder=5)
    ax.scatter([sol1, 0],[1-sol1, 0],[0, 1], c='tab:blue', marker='.', depthshade=False, zorder=10)

    # Partial solution 2
    sol2 = 0.52
    sol2Line, = ax.plot([0, 1],[sol2, 0],[1-sol2, 0], linestyle='--', c='tab:orange', zorder=5)
    ax.scatter([0, 1],[sol2, 0],[1-sol2, 0], c='tab:orange', marker='.', depthshade=False, zorder=10)

    # Solution point
    solPoint = ax.scatter([0.14104882], [0.44665461], [0.41229656], c='tab:green', marker='.', depthshade=False, zorder=15)

    solutionSurfaceFakeLine = mpl.lines.Line2D([0],[0], linestyle="none", c='tab:grey', marker = 'o')
    ax.legend([solutionSurfaceFakeLine, sol1Line, sol2Line, solPoint], 
              [r'$\sum_{i=1}^n\omega_i=1$', 
               r'Solution Subspace $\mathcal{S}_1(\vec{\gamma})$', 
               r'Solution Subspace $\mathcal{S}_2(\vec{\gamma})$', 
               r'FCI Solution $\vec{\omega}$'],
               loc='center left',
               bbox_to_anchor=(1.15, 0.5))

    ax.xaxis.set_ticks([0,0.5,1])
    ax.yaxis.set_ticks([0,0.5,1])
    ax.zaxis.set_ticks([0,0.5,1])

    if SAVE_FIGS:
        plt.savefig('figures/cloud_fusion_secfci_partial_sol_intersection.pdf')
    else:
        plt.show()
    plt.close()


two_sen()
# p_sol_1()
# p_sol_2()
# p_sol_intersection()