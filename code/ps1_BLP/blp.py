#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################
# Description: BLP
# Date: 2024-01-21
# Author: Alex Lan Xi
###############################################

from os.path import join
import pandas as pd
import numpy as np
import pyblp
import os

psdir = os.getcwd().split("/code")[0]
rawdir = join(psdir, "data/ps1_BLP")
product_data = pd.read_csv(join(rawdir, "ps1_ex4.csv"))
product_data = product_data.rename(columns = {"p":"prices", "market":"market_ids"})
product_data.rename(columns={f'z{i}': f'demand_instruments{i-1}' for i in range(1, 7)}, inplace=True)

# generate agent data
agent_data = pd.DataFrame()
for i in range(1,101):
    agent_mat = np.random.multivariate_normal([0,0], np.identity(2), size = 50)
    agent_mat = pd.DataFrame(agent_mat, columns = ["nodes0", "nodes1"])
    agent_mat["market_ids"] = i
    agent_mat["weights"] = 0.02
    agent_data = pd.concat([agent_data, agent_mat], ignore_index=True)

pyblp.options.digits = 2
pyblp.options.verbose = False

X1_formulation = pyblp.Formulation('0 + prices + x')
X2_formulation = pyblp.Formulation('0 + prices + x')
product_formulations = (X1_formulation, X2_formulation)

blp_problem = pyblp.Problem(product_formulations, product_data, None, agent_data)

blp_results1 = blp_problem.solve(
    np.identity(2),
    optimization=pyblp.Optimization('bfgs'),
    method='1s'
)

blp_results2 = blp_problem.solve(
    np.random.rand(2, 2),
    optimization=pyblp.Optimization('bfgs'),
    method='1s'
)

# Nonlinear Coefficient Estimates (Robust SEs in Parentheses):
# =========================================================================
# Sigma:    prices        x       |  Sigma Squared:    prices        x     
# ------  ----------  ----------  |  --------------  ----------  ----------
# prices   -4.5E+00               |      prices       +2.0E+01    +1.1E-01 
#         (+5.1E+00)              |                  (+4.6E+01)  (+1.6E+01)
#                                 |                                        
#   x      -2.4E-02    -4.4E-02   |        x          +1.1E-01    +2.5E-03 
#         (+3.5E+00)  (+2.7E+00)  |                  (+1.6E+01)  (+3.4E-01)
# =========================================================================

# Beta Estimates (Robust SEs in Parentheses):
# ======================
#   prices        x     
# ----------  ----------
#  -1.5E+00    -1.7E-01 
# (+2.0E+00)  (+4.6E-01)
# ======================
