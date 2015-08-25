"""
Outputs


"""
from class_three import deforestation_growth
from numpy import linspace, mean, std
from numpy.random import lognormal, normal
import csv

"""
setting parameter values
"""

"Preference and technology parameters"
beta = 0.9		        # discount rate
theta = 2.0	            # utility parameter
TFP = 0.6	            # TFP
alpha = 0.45	        # output share capital
gamma = 0.15		    # output share land
costEXP = 2.0	        # exponent on cost function
costK = 0.6 		    # constant on cost function
delta = 0.11	        # capital depreciation rate
"Shock parameters"
sigma_c = 0.675					# std dev cost shock
mu_c = -(sigma_c**2)/2.0		# mean cost shock = 1
dist_c = lognormal		    # distribution cost shock
mu_l = 0				    # mean land shock
sigma_l = 0.000001		    # std dev cost shock
dist_l = normal			    # distribution land shock
mu_y = -(sigma_c**2)/2.0	    # mean output shock = 1
sigma_y = 0.1			    # std dev ouput shock
dist_y = lognormal		    # distribution output shock
"Conversion parameters"
landtoemissions = 10000 * 100 * 150 * 3.67   # land in CO2e
paymentstodollars = 1e9 * 709.2			# output in USD2010
"Grid paramaters"
gridSize = 10
minIncome, maxIncome = 1.0, 6.0
gridIncome = linspace(minIncome**(1/3.0), maxIncome**(1/3.0), gridSize)**3.0
minLand, maxLand = 4.75, 7.0
gridLand = linspace(minLand**(1/1.0), maxLand**(1/1.0), gridSize)**1.0
baseGridShock = dist_c(mu_c, sigma_c, 1e7)	    # take a large sample
n = 1e7 / 20
gridShock = baseGridShock[::n]		            # take each nth element
gridShock.sort()
minShock, maxShock = min(baseGridShock), max(baseGridShock)
gridShock[0] = minShock
gridShock[-1] = maxShock		                # make grid wide as poss

initialState = (1.44, 4.9246, 1)
tol = 1e-5					    # tolerance for fvi
sampleSize = 1000
# Experiment One: Comparing policies
# Baseline
taxRate = 0
adPolicyFlag = None
grandfather = 0
"Collecting parameters"
params = (
    beta, TFP, theta, alpha, gamma, costEXP, costK, delta
    )
shockparams = (
    mu_c, sigma_c, dist_c, mu_l, sigma_l, dist_l, mu_y, sigma_y, dist_y
    )
gridparams = (
    gridLand, gridShock, gridIncome, initialState, tol,
    landtoemissions, paymentstodollars
    )
policyparams = (taxRate, adPolicyFlag, grandfather)

base_instance = deforestation_growth(
    params, shockparams, gridparams, policyparams
    )
base_instance.fittedValueIteration()
base_instance.transfer = 0
base_policy = base_instance.policy
base_valueFunction = base_instance.newvaluefunction
base_instance.sampleTaker(sampleSize, 30, base_policy)
base_deforestation = \
    base_instance.landsequence[1:] - base_instance.landsequence[:-1]
mean_base_deforestation = mean(base_deforestation)
std_base_deforestation = std(base_deforestation)
max_base_deforestation = max(base_deforestation)
mean_landholdings = mean(base_instance.sample_land)

# Deforestation tax
print 'Deforestation tax'
taxRate = 0.0077645
adPolicyFlag = 'defoTax'
grandfather = 0.0
policyparams = (taxRate, adPolicyFlag, grandfather)
defo_instance = deforestation_growth(
    params, shockparams, gridparams, policyparams
    )
defo_instance.fittedValueIteration()
defo_instance.transferFinder(base_valueFunction)
defo_transfer = defo_instance.transfer
defo_instance.sampleTaker(sampleSize, 30, base_policy)
defo_vf = defo_instance.newvaluefunction
defo_pf = defo_instance.policy
defo_landsequence = defo_instance.landsequence

output_table = []
output_table.append(
    ['Policy', 'Transfer', 'Payments', 'Avoided defo', 'Cost', 'Net cost',
        '\\\\']
    )
output_table.append(
    ['Deforestation tax', round(defo_transfer / 1e9, 4), round(
        mean(defo_instance.sample_payments) / 1e9, 4), round(mean(
            defo_instance.sample_avoidedDeforestation)/1e6, 4), round(mean(
                defo_instance.sample_costperco2), 4), round(mean(
                    defo_instance.sample_netcostperco2), 4), '\\\\']
    )
print 'Deforestation tax'
print output_table[0]
print output_table[-1]
print 'mean baseline: ', mean_landholdings
print 'mean tax landholdings: ', mean(defo_instance.sample_land)


# Avoided deforestation - quasi optimal
print 'Avoided deforestation'
adPolicyFlag = 'adSub'
grandfather = defo_transfer * (1-beta) / taxRate
print 'Transfer: ', defo_transfer
print 'Optimal gf: ', grandfather
print 'Mean defo: ', mean_base_deforestation
policyparams = (taxRate, adPolicyFlag, grandfather)
defo_instance = deforestation_growth(
    params, shockparams, gridparams, policyparams
    )
defo_instance.fittedValueIteration()
defo_instance.transferFinder(base_valueFunction)
defo_transfer = defo_instance.transfer
defo_instance.sampleTaker(sampleSize, 30, base_policy)
defo_vf = defo_instance.newvaluefunction
defo_pf = defo_instance.policy
defo_landsequence = defo_instance.landsequence

output_table.append(
    ['AD: T*(1-beta)/tau', round(defo_transfer, 4), round(
        mean(defo_instance.sample_payments) / 1e9, 4), round(mean(
            defo_instance.sample_avoidedDeforestation)/1e6, 4), round(mean(
                defo_instance.sample_costperco2), 4), round(mean(
                    defo_instance.sample_netcostperco2), 4), '\\\\']
    )

print 'Avoided deforestation - optimal Q linear baseline'
print output_table[0]
print output_table[-1]
# Avoided deforestation - mean defo as base
print 'Mean deforestation as base'
grandfather = mean_base_deforestation
policyparams = (taxRate, adPolicyFlag, grandfather)
defo_instance = deforestation_growth(
    params, shockparams, gridparams, policyparams
    )
defo_instance.fittedValueIteration()
defo_instance.transferFinder(base_valueFunction)
defo_transfer = defo_instance.transfer
defo_instance.sampleTaker(sampleSize, 30, base_policy)
defo_vf = defo_instance.newvaluefunction
defo_pf = defo_instance.policy
defo_landsequence = defo_instance.landsequence

output_table.append(
    ['AD: historical', round(defo_transfer, 4), round(
        mean(defo_instance.sample_payments) / 1e9, 4), round(mean(
            defo_instance.sample_avoidedDeforestation)/1e6, 4), round(mean(
                defo_instance.sample_costperco2), 4), round(mean(
                    defo_instance.sample_netcostperco2), 4), '\\\\']
    )

print 'Avoided deforestation - historial baseline'
print output_table[0]
print output_table[-1]
# OUTPUT ONE
with open('output_table_policies.csv', 'wb') as f:
    writer = csv.writer(f, delimiter='&')
    writer.writerows(output_table)
