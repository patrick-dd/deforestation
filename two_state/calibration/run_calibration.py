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
from numpy import linspace, mean, std, array
from numpy.random import lognormal
import matplotlib.pyplot as plt
import csv

# setting parameter values
beta = 0.9				    # discount rate
TFP = 0.75				    # ag TFP
gamma = 0.2                 # land share income
theta = 1.0                 # utility parameter
costEXP = 2.0			    # exponent on cost function
costK = 3.675                   # multiplicative constant on cost function
sigma = 0.69			            # standard deviation on cost shock
mu = -(sigma**2)/2.0        # mean on cost shock = 1 with a lognormal dist
landtoemissions = 10000 * 100 * 150 * 3.67	    # land units in emissions
paymentstodollars = 1e9 * 60.68	 # output units in billion usd2010

# setting numerical values
tol = 1e-5              # tolerance for fitted value iteration
minLand, maxLand = 4.75, 8.0
gridSize = 100
gridLand = linspace(minLand**(1/1.0), maxLand**(1/1.0), gridSize)**1.0
# making shock space the same shape as the distribution of shocks
baseGridShocks = lognormal(mu, sigma, 1e7)
n = 1e7 / 50
gridShocks = baseGridShocks[::n]
gridShocks.sort()
# ensure that the shock space is wide enough
minShocks, maxShocks = min(baseGridShocks), max(baseGridShocks)
gridShocks[0] = minShocks
gridShocks[-1] = maxShocks
initialState = (4.9246, 1)
params = (
    beta, TFP, gamma, theta, costEXP, costK, mu, sigma,
    landtoemissions, paymentstodollars)
gridparams = (gridLand, gridShocks, initialState, tol)

base_instance = defo_calibration(params, gridparams)
base_instance.fittedValueIteration()
base_instance.transfer = 0
base_policy = base_instance.policy
base_valueFunction = base_instance.newvaluefunction
base_instance.sampleTaker(10000, 6, base_policy)
base_instance.empiricalDistributionPlot(base_instance.sample_land)
mean_landholdings = mean(base_instance.sample_land)

# Getting historical data
state_variable_data_labels = []
state_variable_data = []
count = 1
with open('statevarvalues.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        state_variable_data_labels.append(row[0])
        data_row = []
        if count < 3:
            for element in row[1:]:
                data_row.append(float(element))
        else:
            for element in row[1:]:
                data_row.append(int(element))
        state_variable_data.append(data_row)
        count += 1

hist_ag_land = state_variable_data[0]
hist_ag_gdp = state_variable_data[1]
hist_years = state_variable_data[2]
hist_deforestation = array(hist_ag_land[1:]) - array(hist_ag_land[:-1])

plt.plot(hist_years, hist_ag_land, label='Historical')
plt.plot(hist_years, base_instance.landsequence, label='Model')
plt.legend()
plt.show()

print 'Percentage deviation from historical land accumulation: ',\
    100*(mean_landholdings-hist_ag_land[-1])/(hist_ag_land[-1]-hist_ag_land[0])
print 'Mean landholdings: ', mean_landholdings
print ''
print 'Model deforestation SD:      ', mean(base_instance.sample_d_sd)
print 'Historical deforestation SD: ', std(hist_deforestation)

sd_line = [std(hist_deforestation)]*len(base_instance.sample_d_sd)

plt.plot(base_instance.sample_d_sd)
plt.plot(sd_line)
plt.show()
