"""

Outputs

1. Table.
Columns: AD, Land, Defo
Rows: PV transfer, Avoided deforestation, Cost effectiveness, Received payments
2. Land policy 3 dimensional plot
3. CDF of land policy given a particular landholdings

Calibration values

Land is in 1000 square kilometres
Output is in Billion of 2010 USD
Tax rates in 2010USD 66.6 per tonne carbon ~ 2010 USD 18 per tonne CO2


"""
from class_defo_calibration import defo_calibration
from numpy import linspace, mean
from numpy.random import lognormal
import csv

# setting parameter values
beta = 0.9				    # discount rate
TFP = 30				    # ag TFP
gamma = 0.2                 # land share income
theta = 1.0                 # utility parameter
costEXP = 2.0			    # exponent on cost function
costK = 1.5                 # multiplicative constant on cost function
sigma = 1.0			    # standard deviation on cost shock
mu = -(sigma**2)/2.0        # mean on cost shock = 1 with a lognormal dist
landtoemissions = 1000 * 100 * 150 * 3.67	    # land units in emissions
paymentstodollars = 1e9	 # output units in billion usd2010

# setting numerical values
tol = 1e-3             # tolerance for fitted value iteration
minLand, maxLand = 47.5, 80
gridSize = 3
gridLand = linspace(minLand**(1/1.0), maxLand**(1/1.0), gridSize)**1.0
# making shock space the same shape as the distribution of shocks
baseGridShocks = lognormal(mu, sigma, 1e7)
n = 1e7 / 3
gridShocks = baseGridShocks[::n]
gridShocks.sort()
# ensure that the shock space is wide enough
minShocks, maxShocks = min(baseGridShocks), max(baseGridShocks)
gridShocks[0] = minShocks
gridShocks[-1] = maxShocks
initialState = (49.246, 1)


theta_list = linspace(0.5, 2.0, 10)
sigma_list = linspace(0.5, 2.0, 10)
costK_list = linspace(0.5, 2.5, 10)

landholdings_grid = []
defo_sd_grid = []
for theta in theta_list:
    outer_lh = []
    outer_sd = []
    for cK in costK_list:
        inner_lh = []
        inner_sd = []
        for s in sigma_list:
            params = (
                beta, TFP, gamma, theta, costEXP, costK, mu, sigma,
                landtoemissions, paymentstodollars)
            gridparams = (gridLand, gridShocks, initialState, tol)
            base_instance = defo_calibration(params, gridparams)
            base_instance.fittedValueIteration()
            base_instance.transfer = 0
            base_policy = base_instance.policy
            base_valueFunction = base_instance.newvaluefunction
            base_instance.sampleTaker(1000, 6, base_policy)
            mean_landholdings = mean(base_instance.sample_land)
            mean_defo_sd = mean(base_instance.sample_d_sd)
            inner_lh.append(mean_landholdings)
            inner_sd.append(mean_defo_sd)
        outer_lh.append(inner_lh)
        inner_sd.append(inner_sd)
    landholdings_grid.append(inner_lh)
    defo_sd_grid.append(inner_sd)

with open('landholdings.csv', 'wb') as csvfile:
    datawriter = csv.writer(csvfile, delimiter=',')
    for i in range(len(theta_list)):
        for (cK, row) in zip(costK_list, landholdings_grid):
            row = [theta_list[i], cK] + row
            datawriter.writerow(row)

with open('defo_sd.csv', 'wb') as csvfile:
    datawriter = csv.writer(csvfile, delimiter=',')
    for i in range(len(theta_list)):
        for (cK, row) in zip(costK_list, defo_sd_grid):
            row = [theta_list[i], cK] + row
            datawriter.writerow(row)
