"""
Outputs


"""
from class_three import deforestation_growth
from numpy import linspace, mean, std, array, nanmean
from numpy.random import lognormal, normal
import matplotlib.pyplot as plt
import csv
import pickle

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
costK = 0.6 		# constant on cost function
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
landtoemissions = 1000 * 100 * 150 * 3.67   # land in CO2e
paymentstodollars = 1e9 * 536.63			# output in USD2010
"Grid paramaters"
gridSize = 10
minIncome, maxIncome = 1.0, 6.0
gridIncome = linspace(minIncome**(1/3.0), maxIncome**(1/3.0), gridSize)**3.0
minLand, maxLand = 4.75, 7.0
gridLand = linspace(minLand**(1/1.0), maxLand**(1/1.0), gridSize)**1.0
baseGridShock = dist_c(mu_c, sigma_c, 1e7)	    # take a large sample
n = 1e7 / 5
gridShock = baseGridShock[::n]		            # take each nth element
gridShock.sort()
minShock, maxShock = min(baseGridShock), max(baseGridShock)
gridShock[0] = minShock
gridShock[-1] = maxShock		                # make grid wide as poss

initialState = (1.44, 4.9246, 1)
tol = 1e-3					    # tolerance for fvi
"Policy parameters"
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

base_policy = pickle.load(open('base_policy.p', 'rb'))
base_valueFunction = pickle.load(open('base_valueFunction.p', 'rb'))

base_instance = deforestation_growth(
    params, shockparams, gridparams, policyparams
    )
base_instance.fittedValueIteration()
#    initialV=base_valueFunction, initialP=base_policy
#    )
base_instance.transfer = 0
base_policy = base_instance.policy
base_valueFunction = base_instance.newvaluefunction
base_instance.sampleTaker(1000, 6, base_policy)
base_deforestation = \
    base_instance.landsequence[1:] - base_instance.landsequence[:-1]
mean_base_deforestation = mean(base_deforestation)
std_base_deforestation = std(base_deforestation)
max_base_deforestation = max(base_deforestation)
mean_landholdings = mean(base_instance.sample_land)
nanmean_landholdings = nanmean(base_instance.sample_land)
mean_income = mean(base_instance.sample_income)
nanmean_income = nanmean(base_instance.sample_income)

pickle.dump(base_valueFunction, open('base_valueFunction.p', 'wb'))
pickle.dump(base_policy, open('base_policy.p', 'wb'))

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
hist_income = state_variable_data[1]
hist_years = state_variable_data[2]
hist_deforestation = array(hist_ag_land[1:]) - array(hist_ag_land[:-1])

plt.title('Land')
plt.plot(hist_years, hist_ag_land, label='Historical')
plt.plot(hist_years, base_instance.landsequence, label='Model')
plt.legend()
plt.show()

plt.title('Income')
plt.plot(hist_years, hist_income, label='Historical')
plt.plot(hist_years, base_instance.incomesequence, label='Model')
plt.legend()
plt.show()

print 'Percentage deviation from historical land accumulation: ',\
    100*(mean_landholdings-hist_ag_land[-1])/(hist_ag_land[-1]-hist_ag_land[0])
print 'Mean landholdings: ', mean_landholdings
print 'NaNmean landholdings: ', nanmean_landholdings
print 'Historical landholdings: ', hist_ag_land[-1]
print ''
print 'Mean Model deforestation SD:      ', mean(base_instance.sample_d_sd)
print 'NaNmean Model deforestation SD:      ',\
    nanmean(base_instance.sample_d_sd)
print 'Historical deforestation SD: ', std(hist_deforestation)
print ''
print 'Mean income: ', mean_income
print 'NaNmean income: ', nanmean_income
print 'Historical income: ', hist_income[-1]
print ''

sd_line = [std(hist_deforestation)] * len(base_instance.sample_d_sd)

plt.plot(base_instance.sample_d_sd)
plt.plot(sd_line)
plt.show()
