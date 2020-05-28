"""
======
ANANKE
======

Example 1
---------
Solves a 1D control problem.

@author: jasonmeverett
"""

import AnankeC
import matplotlib.pyplot as plt
import time
import ex1_setup as ex1
from ananke.opt import *

RegionFlags = AnankeC.RegionFlags
ObjectiveFlags = AnankeC.ObjectiveFlags

use_cpp = True

t1 = time.time()

# =============================================================================
#                                                               PROBLEM SET UP
# =============================================================================
num_nodes = 100

if use_cpp:
    # =============================================================================
    # =============================================================================
    # Configure Ananke optimizer. This Python class directly inherits the functions
    # of a PyGMO problem() class, and can be used as such in code.
    ao = AnankeC.Ananke_Config()
    tl1 = AnankeC.TrajLeg(num_nodes, 1.0)
    tl1.set_len_X_U(2, 1)
    tl1.set_dynamics(AnankeC.ex1.f, AnankeC.ex1.df, [])
    tl1.add_eq(AnankeC.ex1.g1, AnankeC.ex1.dg1, 2, RegionFlags.FRONT, [])
    tl1.add_eq(AnankeC.ex1.g2, AnankeC.ex1.dg2, 2, RegionFlags.BACK, [])
    tl1.add_ineq(AnankeC.ex1.g3, AnankeC.ex1.dg3, 1, RegionFlags.PATH, [4.5])
    tl1.set_obj(AnankeC.ex1.Jctrl, AnankeC.ex1.dJctrl, ObjectiveFlags.LAGRANGE, [])
    tl1.set_TOF(1.0, 1.0)
    bnds_min = [-100.0, -100.0, -100.0]
    bnds_max = [ 100.0,  100.0,  100.0]
    tl1.set_bounds(bnds_min, bnds_max)

    # Add a trajectory leg.
    ao.add_leg(tl1)
    ao.set_TOF(0.0, 1.0)
    #ao.use_estimate_grad = True
    #ao.est_grad_dt = 1e-8

    X0 = [1.0]
    for ii in range(0, num_nodes):
        X0 = X0 + [float(ii)/float(num_nodes), 1.0, 1.0 - float(ii)/float(num_nodes)] 

    
    AnankeC.set_dv(X0)
    AnankeC.set_ac(ao)
    X, F = AnankeC.optimize(30, 1)
    # =============================================================================
    # =============================================================================

else:
    # =============================================================================
    # =============================================================================
    # Configure Ananke optimizer. This Python class directly inherits the functions
    # of a PyGMO problem() class, and can be used as such in code.
    ao = Ananke_Config()
    tl1 = TrajLeg(num_nodes, 1.0)
    tl1.set_len_X_U(2, 1)
    tl1.set_dynamics(ex1.f, ex1.df, [])
    tl1.add_eq(ex1.g1, ex1.dg1, 2, RegionFlags.FRONT, [])
    tl1.add_eq(ex1.g2, ex1.dg2, 2, RegionFlags.BACK, [])
    tl1.add_ineq(ex1.g3, ex1.dg3, 1, RegionFlags.PATH, [4.5])
    tl1.set_obj(ex1.Jctrl, ex1.dJctrl, ObjectiveFlags.LAGRANGE, [])
    tl1.set_TOF(1.0, 1.0)
    bnds_min = [-100.0, -100.0, -100.0]
    bnds_max = [ 100.0,  100.0,  100.0]
    tl1.set_bounds(bnds_min, bnds_max)

    # Add a trajectory leg.
    ao.add_leg(tl1)
    ao.set_TOF(0.0, 1.0)

    X0 = [1.0]
    for ii in range(0, num_nodes):
        X0 = X0 + [float(ii)/float(num_nodes), 1.0, 1.0 - float(ii)/float(num_nodes)] 

    prob = pg.problem(ao)
    prob.c_tol = 1e-5
    algo = pg.algorithm(pg.nlopt('slsqp'))
    algo.set_verbosity(1)
    algo.extract(pg.nlopt).xtol_rel = 0.0
    algo.extract(pg.nlopt).ftol_rel = 0.0
    algo.extract(pg.nlopt).maxeval = 30

    print(prob)
    pop = pg.population(prob)
    pop.push_back(X0)
    pop = algo.evolve(pop)
    X = pop.champion_x
    F = pop.champion_f
    # =============================================================================
    # =============================================================================

t2 = time.time()
dt = t2 - t1
print("delta: ", t2 - t1)

# Grab first leg data.
outdata = ao.get_array_data(X)[0]

# Plot information.
fig, axs = plt.subplots(3,1,sharex=True,squeeze=True)
axs[0].plot(outdata[:,0],outdata[:,1],marker='*')
axs[1].plot(outdata[:,0],outdata[:,2],marker='*')
axs[2].plot(outdata[:,0],outdata[:,3],marker='*')
axs[0].grid(which='both')
axs[0].minorticks_on()
axs[1].grid(which='both')
axs[1].minorticks_on()
axs[2].grid(which='both')
axs[2].minorticks_on()
axs[0].set_ylabel('Position')
axs[1].set_ylabel('Velocity')
axs[2].set_ylabel('Control')
plt.show()



