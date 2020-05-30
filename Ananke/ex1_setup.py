"""
======
ANANKE
======

Example 1
---------
Solves a 1D control problem.

@author: jasonmeverett
"""

import numpy as np
# import pygmo as pg
import json

# =============================================================================
#                                                      DYNAMICS AND CONSTRAINTS
# =============================================================================

# -----------------------------------------------------------
# System dynamics for legs.
# -----------------------------------------------------------
def f(X, U, T, params):
    dX = np.zeros(2)
    dX[0] = X[1]
    dX[1] = U[0]
    return dX

def df(X, U, T, params):
    dfdX = np.zeros((2,2))
    dfdU = np.zeros((2,1))
    dfdX[0,1] = 1.0
    dfdU[1,0] = 1.0
    return [dfdX, dfdU]

# -----------------------------------------------------------
# Objective function - minimum control
# -----------------------------------------------------------   
def Jctrl(X, U, T, params):
    J = U[0]**2.0
    return J

def dJctrl(X, U, T, params):
    dJdX = np.zeros((1, 2))
    dJdU = np.zeros((1, 1))
    dJdU[0,0] = 2.0 * U[0]
    dJdT = np.zeros((1,1))
    return [dJdX, dJdU, dJdT]

def Jfuel(X, U, T, params):
    J = np.sqrt(U[0]*U[0])
    return J

def dJfuel(X, U, T, params):
    dJdX = np.zeros((1, 2))
    dJdU = np.zeros((1, 1))
    dJdU[0,0] = 2.0 * U[0] * 1 / (2*np.sqrt(U[0]*U[0]))
    dJdT = np.zeros((1,1))
    return [dJdX, dJdU, dJdT]

# -----------------------------------------------------------
# Constraint 1: starting state
# -----------------------------------------------------------
def g1(X, U, T, params):
    g = np.zeros(2)
    g[0] = X[0]
    g[1] = X[1]
    return g

def dg1(X, U, T, params):
    dgdX = np.zeros((2, 2))
    dgdU = np.zeros((2, 1))
    dgdT = np.zeros((2, 1))
    dgdX[0,0] = 1.0
    dgdX[1,1] = 1.0
    return [dgdX, dgdU, dgdT]

# -----------------------------------------------------------
# Constraint 2: ending state
# -----------------------------------------------------------
def g2(X, U, T, params):
    g = np.zeros(2)
    g[0] = X[0] - 1.0
    g[1] = X[1]
    return g

def dg2(X, U, T, params):
    dgdX = np.zeros((2, 2))
    dgdU = np.zeros((2, 1))
    dgdT = np.zeros((2, 1))
    dgdX[0, 0] = 1.0
    dgdX[1, 1] = 1.0
    return [dgdX, dgdU, dgdT]

# -----------------------------------------------------------
# Constraint 3: controls path constraint limit
# -----------------------------------------------------------
def g3(X, U, T, params):
    ulim = params[0]
    g = np.zeros(1)
    g[0] = np.sqrt(U[0]**2.0) - ulim
    return g

def dg3(X, U, T, params):
    dgdX = np.zeros((1, 2))
    dgdU = np.zeros((1, 1))
    dgdT = np.zeros((1, 1))
    dgdU[0, 0] = U[0]/np.sqrt(U[0]**2.0)
    return [dgdX, dgdU, dgdT]



