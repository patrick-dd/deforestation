"""

Calibration values

Land is in 1000 square kilometres
Output is in Billion of 2010 USD
Tax rates in 2010USD 66.6 per tonne carbon ~ 2010 USD 18 per tonne CO2

"""
from class_deforestation import class_defo
from numpy import linspace, mean, std, array
from numpy.random import lognormal, uniform
from scipy import interpolate
import matplotlib.pyplot as plt
from ecdf import ECDF
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

print paymentstodollars/landtoemissions

# setting numerical values
tol = 1e-5             # tolerance for fitted value iteration
minLand, maxLand = 4.75, 8.0
gridSize = 50
gridLand = linspace(minLand**(1/1.0), maxLand**(1/1.0), gridSize)**1.0
# making shock space the same shape as the distribution of shocks
baseGridShocks = lognormal(mu, sigma, 1e7)
n = 1e7 / 30
gridShocks = baseGridShocks[::n]
gridShocks.sort()
sampleSize = 10000
# ensure that the shock space is wide enough
minShocks, maxShocks = min(baseGridShocks), max(baseGridShocks)
gridShocks[0] = minShocks
gridShocks[-1] = maxShocks
initialState = (4.9246, 1)
params = (
    beta, TFP, gamma, theta, costEXP, costK, mu, sigma,
    landtoemissions, paymentstodollars)
gridparams = (gridLand, gridShocks, initialState, tol)

# Experiment One: Comparing policies
# Baseline
taxRate = 0
adPolicyFlag = None
grandfather = 0.0
policyparams = (taxRate, adPolicyFlag, grandfather)
base_instance = class_defo(params, gridparams, policyparams)
base_instance.fittedValueIteration()
base_instance.transfer = 0
base_policy = base_instance.policy
base_valueFunction = base_instance.newvaluefunction
base_instance.sampleTaker(sampleSize, 30, base_policy)
base_deforestation = base_instance.landsequence[1:] -\
    base_instance.landsequence[:-1]
mean_base_deforestation = mean(base_deforestation)
std_base_deforestation = std(base_deforestation)
max_base_deforestation = max(base_deforestation)

baselines = linspace(
    mean_base_deforestation - std_base_deforestation,
    mean_base_deforestation + std_base_deforestation,
    10)

# Deforestation tax
print 'Deforestation tax'
taxRate = 1e-5 * paymentstodollars / landtoemissions
adPolicyFlag = 'defoTax'
grandfather = 0.0
policyparams = (taxRate, adPolicyFlag, grandfather)
defo_instance = class_defo(params, gridparams, policyparams)
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
        '\\\\'])
output_table.append(
    ['Deforestation tax',
        round(defo_transfer, 4) * paymentstodollars / 1e9,
        round(mean(defo_instance.sample_payments) / 1e9, 4),
        round(mean(defo_instance.sample_avoidedDeforestation)/1e6, 4),
        round(mean(defo_instance.sample_costperco2), 4),
        round(mean(defo_instance.sample_netcostperco2), 4),
        '\\\\'])

# Avoided deforestation
print 'Avoided deforestation'
adPolicyFlag = 'adSub'
grandfather = defo_transfer * (1 - beta) / taxRate
print 'Transfer: ', defo_transfer
print 'Optimal gf: ', grandfather
print 'Mean defo: ', mean_base_deforestation
policyparams = (taxRate, adPolicyFlag, grandfather)
defo_instance = class_defo(params, gridparams, policyparams)
defo_instance.fittedValueIteration()
defo_instance.transferFinder(base_valueFunction)
defo_transfer = defo_instance.transfer
defo_instance.sampleTaker(sampleSize, 30, base_policy)
defo_vf = defo_instance.newvaluefunction
defo_pf = defo_instance.policy
defo_landsequence = defo_instance.landsequence

output_table.append(
    ['AD: optimal',
        round(defo_transfer, 4) * paymentstodollars / 1e9,
        round(mean(defo_instance.sample_payments) / 1e9, 4),
        round(mean(defo_instance.sample_avoidedDeforestation)/1e6, 4),
        round(mean(defo_instance.sample_costperco2), 4),
        round(mean(defo_instance.sample_netcostperco2), 4),
        '\\\\'])

print 'Mean deforestation as base'
grandfather = mean_base_deforestation
policyparams = (taxRate, adPolicyFlag, grandfather)
defo_instance = class_defo(params, gridparams, policyparams)
defo_instance.fittedValueIteration()
defo_instance.transferFinder(base_valueFunction)
defo_transfer = defo_instance.transfer
defo_instance.sampleTaker(sampleSize, 30, base_policy)
defo_vf = defo_instance.newvaluefunction
defo_ad_pf = defo_instance.policy
defo_landsequence = defo_instance.landsequence

output_table.append(
    ['AD: historical',
        round(defo_transfer, 4) * paymentstodollars / 1e9,
        round(mean(defo_instance.sample_payments) / 1e9, 4),
        round(mean(defo_instance.sample_avoidedDeforestation)/1e6, 4),
        round(mean(defo_instance.sample_costperco2), 4),
        round(mean(defo_instance.sample_netcostperco2), 4),
        '\\\\'])

# OUTPUT ONE
with open('output_table_policies.csv', 'wb') as f:
    writer = csv.writer(f, delimiter='&')
    writer.writerows(output_table)

# Experiment Two: Varying the baseline
output_table = []
for rl in baselines:
    grandfather = rl
    print 'Grandfather less average deforestation: ', grandfather - \
        mean_base_deforestation
    print 'Grandfather: ', grandfather
    policyparams = (taxRate, adPolicyFlag, grandfather)
    ad_instance_neg = class_defo(params, gridparams, policyparams)
    ad_instance_neg.fittedValueIteration()
    ad_instance_neg.transferFinder(base_valueFunction)
    ad_instance_neg.sampleTaker(sampleSize, 30, base_policy)
    ad_vf_neg = ad_instance_neg.newvaluefunction
    ad_pf_neg = ad_instance_neg.policy
    ad_transfer_neg = ad_instance_neg.transfer
    ad_landsequence_neg = ad_instance_neg.landsequence
    output_table.append(
        [round(rl, 4),
            round(ad_transfer_neg, 4) * paymentstodollars / 1e9,
            round(mean(ad_instance_neg.sample_payments) / 1e9, 4),
            round(mean(ad_instance_neg.sample_avoidedDeforestation)/1e6, 4),
            round(mean(ad_instance_neg.sample_costperco2), 4),
            '\\\\'])

# OUTPUT FOUR
with open('output_table_baselines.csv', 'wb') as f:
    writer = csv.writer(f, delimiter='&')
    writer.writerows(output_table)

# From experiment one.
# OUTPUT TWO
ad_deforestation = defo_ad_pf(defo_instance.state) -\
    defo_instance.gridLand.ravel()
for t in range(len(ad_deforestation)):
    if abs(ad_deforestation[t]) < tol:
        ad_deforestation[t] = 0
ad_deforestation = interpolate.LinearNDInterpolator(defo_instance.state,
                                                    ad_deforestation)
defo_instance.threeDimPlot(ad_deforestation)

# OUTPUT THREE
initial_land = initialState[0]
costshocks = lognormal(mu, sigma, sampleSize)
defosequence = []
for t in range(sampleSize):
    defosequence.append(defo_ad_pf(initial_land, costshocks[t]) - initial_land)

defosequence_ad = array(defosequence) * landtoemissions / 1e6

defosequence = []
for t in range(sampleSize):
    defosequence.append(defo_pf(initial_land, costshocks[t]) - initial_land)

defosequence_defo = array(defosequence) * landtoemissions / 1e6

lower = min(min(defosequence_defo), min(defosequence_ad))
upper = max(max(defosequence_defo), max(defosequence_ad))
plotDomain = linspace(lower, upper, len(costshocks))
empiricalCDF = ECDF([uniform(0, 1) for i in range(len(costshocks))])
empiricalCDF.observations = defosequence_ad
obs = []
for j in range(sampleSize):
    obs.append(empiricalCDF(plotDomain[j]))
ecdf_sample_ad = array(obs)
plt.plot(plotDomain, ecdf_sample_ad, label='AD')
empiricalCDF.observations = defosequence_defo
obs = []
for j in range(sampleSize):
    obs.append(empiricalCDF(plotDomain[j]))
ecdf_sample_defo = array(obs)
plt.plot(plotDomain, ecdf_sample_defo, label='Defo')
plt.legend()
plt.show()

plt.plot(defo_ad_pf(gridLand, 1.0) - gridLand, label='AD')
plt.plot(defo_pf(gridLand, 1.0) - gridLand, label='Defo')
plt.legend()
plt.show()
