"""
======
ANANKE
======

Example 1
---------
Solves a 3D lander control problem

@author: jasonmeverett
"""

import AnankeC
from ananke.orbit import *
from ananke.frames import *
from ananke.util import *
from ananke.planets import *
import lander_2ph_setup as lo
from scipy.linalg import norm
import numpy as np
import json

RegionFlags = AnankeC.RegionFlags
ObjectiveFlags = AnankeC.ObjectiveFlags

# =============================================================================
#                                                               PROBLEM SET UP
# =============================================================================

# Parameters
mu = 4902799000000.0
R_eq = 1738000.0
Omega = 0.0000026616665
Tmax = 100000.0
isp = 450.0
etaLB = 0.2
etaUB = 0.9

# Configure starting state
m0 = 30000.0
alta = 100000.0
altp = 15000.0
ra = alta + R_eq
rp = altp + R_eq
sma = 0.5*(ra+rp)
ecc = (ra-rp)/(ra+rp)
inc = 0.0
raan = 0.0
argper = 0.0
ta = -23.0
r0_I,v0_I = elts_to_rv(sma,ecc,inc,raan,argper,ta,mu,degrees=True)

LSlat = 0.0
LSlon = 0.0
LSalt = 0.0
rf = [200.0, 0.0, 0.0]
vf = [-15.0, 0.0, 0.0]
R_PF_UEN = Rot_PF_UEN(LSlon, LSlat, False)
R_UEN_PF = R_PF_UEN.inv()
parLand = [Omega, R_eq, LSlat, LSlon, LSalt] + rf + vf + np.reshape(R_UEN_PF.as_dcm(), (9,)).tolist()

# Configure Ananke optimizer. This Python class directly inherits the functions
# of a PyGMO problem() class, and can be used as such in code.
ao = AnankeC.Ananke_Config()

# Coasting leg
nns = [10, 20]
nn1 = nns[0]
tl1 = AnankeC.TrajLeg(nn1, 100.0)
tl1.set_len_X_U(7, 4)
tl1.set_dynamics(lo.f, lo.df, [mu, 0.0, isp])
tl1.add_eq(lo.g_X0, lo.dg_X0, 7, RegionFlags.FRONT, (r0_I.tolist() + v0_I.tolist() + [m0]))
tl1.add_eq(lo.g_conU, lo.dg_conU, 1, RegionFlags.PATH, [])
tl1.add_ineq(lo.g_conEtaLB, lo.dg_conEtaLB, 1, RegionFlags.PATH, [0.0])
tl1.add_ineq(lo.g_conEtaUB, lo.dg_conEtaUB, 1, RegionFlags.PATH, [0.0])
tl1.set_TOF(120.0, 1200.0)
bnds_min = 11 * [-2000000]
bnds_max = 11 * [ 2000000]
tl1.set_bounds(bnds_min, bnds_max)

# Configure a trajectory leg.
nn2 = nns[1]
tl2 = AnankeC.TrajLeg(nn2, 300.0)
tl2.set_len_X_U(7, 4)
tl2.set_dynamics(lo.f, lo.df, [mu, Tmax, isp])
tl2.set_obj(lo.Jfuel, lo.dJfuel, ObjectiveFlags.LAGRANGE, [Tmax])
# tl2.set_obj(lo.Jctrl, lo.dJctrl, ObjectiveFlags.LAGRANGE, [Tmax])
tl2.add_eq(lo.g_Xf, lo.dg_Xf, 6, RegionFlags.BACK, parLand)
tl2.add_eq(lo.g_conU, lo.dg_conU, 1, RegionFlags.PATH, [])
tl1.add_ineq(lo.g_alt, lo.dg_alt, 1, RegionFlags.PATH, [R_eq])
tl2.add_ineq(lo.g_conEtaLB, lo.dg_conEtaLB, 1, RegionFlags.PATH, [etaLB])
tl2.add_ineq(lo.g_conEtaUB, lo.dg_conEtaUB, 1, RegionFlags.PATH, [etaUB])
tl2.set_TOF(200.0, 1200.0)
bnds_min = 11 * [-2000000]
bnds_max = 11 * [ 2000000]
tl2.set_bounds(bnds_min, bnds_max)

# Add a trajectory leg.
ao.add_leg(tl1)
ao.add_leg(tl2)
ao.add_leg_link(0, 1, lo.l_12, lo.dl_12, 7, [])
ao.set_TOF(100.0, 2500.0)

## set up initial guess.
#x0 = np.array([r0_I[0], r0_I[1], r0_I[2], v0_I[0], v0_I[1], v0_I[2], m0, 0.0, -1.0, 0.0, 0.3])
#xf = [R_eq, 0, 0, 0, 0, 0, 0.5*x0[6], 0.0, -1.0, 0.0, 0.2]
#xinit = [500]
#dt = 500 / (nns[0]-1)
#for ii in range(0, nn1):
#    dX0 = lo.f(x0[0:7], x0[7:11], 0.0, [mu, 0.0, isp])
#    x0[0:7] = x0[0:7] + dX0[0:7]*dt
#    xinit = xinit + x0.tolist()
#xinit = xinit + [500]
#dt = 500 / (nns[1]-1)
#for ii in range(0, nn2):
#    dX0 = lo.f(x0[0:7], x0[7:11], 0.0, [mu, Tmax, isp])
#    x0[0:7] = x0[0:7] + dX0[0:7]*dt
#    xinit = xinit + x0.tolist()
#x1 = xinit

outold = np.load('champion_out.npy', allow_pickle=True)
x1 = []
T0 = 0.0
outNew = []
for ii,leg in enumerate(outold):
    legNew = np.zeros((nns[ii], leg.shape[1]))
    x1 = x1 + [leg[-1,0] - leg[0,0]]
    for jj in range(0, leg.shape[1]):
        legNew[:,jj] = np.linspace(leg[0,jj],leg[-1,jj],num=nns[ii])
    for jj in range(0, legNew.shape[0]):
        x1 = x1 + legNew[jj,1:].tolist()
    outNew.append(legNew)

AnankeC.set_dv(x1)
AnankeC.set_ac(ao)
aa = 1
X, F = AnankeC.optimize(15000, 50, 1e-4, 1e6)
bb = 2
# Grab first leg data.
outdata = ao.get_array_data(X)
np.save('champion_out', outdata, allow_pickle = True)

# Plot information.
fig, axs = plt.subplots(3,1,sharex=True,squeeze=True)
for leg in outdata:
    axs[0].plot(leg[:,0],leg[:,7],marker='*')
    axs[1].plot(leg[:,0],norm(leg[:,1:4],axis=1) - R_eq,marker='*')
    axs[2].plot(leg[:,0],leg[:,11],marker='*')
axs[0].grid(which='both')
axs[0].minorticks_on()
axs[1].grid(which='both')
axs[1].minorticks_on()
axs[2].grid(which='both')
axs[2].minorticks_on()
axs[0].set_ylabel('Mass')
axs[1].set_ylabel('Altitude')
axs[2].set_ylabel('Throttle')
plt.suptitle("Fuel Optimal Landing Trajectory")
plt.show()




