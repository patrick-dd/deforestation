"""

Old objective: To see if in a deterministic model we can get the equivalence result 
Old issue: [SOLVED] that the land tax model is resulting in more deforestation
Solved with a larger grid size

Current objective: Get welfare equalising transfers for land and defo taxes with stochasticity.
Hypothesis. Both as cost effective as each other.

Next objective. Baseline testing.
Baselines
    1. Historical
    2. Projected
    3. Zero transfer
___________
A note on the units used in this analysis:

	Income:		    Billion 2010 USD 
	Land:		    1,000 square kilometers
	Policy rates:   USD 100 K per ha 

Some adjustment factors.
1 IDR_2000 = 27816/10000 = 2.78 IDR_2010
1 USD_20000701 = 8735 IDR_20000701
1 USD_20100701 = 9070 IDR_20100701
1 USD_2000 = 1.23 USD_2010
1 tC = 3.67 tC0_2
150tC/ha in Indonesia (Saatch et al PNAS 2011)
100 ha in sq km

So, in 2010 USD
	Policy rates:   ( 1 bn USD_2010 / 100000 ha ) / 150 tC/ha
					66.667 USD_2010 per tonne carbon
					18.165 USD_2010 per tonne carbon dioxide (EU ETS)

So, a policy rate of 0.05505 ~ 1 USD_2010 per t CO_2

"""
from class_defo_stochastic import defo_dual
from numpy import linspace, array, mean
from scipy import interpolate
from numpy.random import lognormal
import pickle
import matplotlib.pyplot as plt
import csv

## setting parameter values
beta	= 0.9			# discount rate
TFP	    = 1.4		    # national TFP 
gamma   = 1.0           # land share income
costEXP	= 2.0			# exponent on cost function
costK	= 15			# multiplicative constant on cost function
sigma   = 0.05           # standard deviation on deforestation error
mu      = -(sigma**2)/2.0             # mean on deforestation error



## setting numerical values
tol	        = 1e-5      # tolerance for fitted value iteration
minLand, maxLand 	= 47.5,60 
gridSize 	= 10
gridLand	= linspace( minLand**(1/5.0), maxLand**(1/5.0), gridSize )**5.0	
## making shock space the same shape as the distribution of shocks
baseGridShocks = lognormal(mu,sigma,1e7)
n = 1e7 / 30
gridShocks = baseGridShocks[::n]
gridShocks.sort()
# to ensure that the shock space is wide enough
minShocks, maxShocks = min(baseGridShocks), max(baseGridShocks)
gridShocks[0] = minShocks
gridShocks[-1] = maxShocks


initialState= ( 49.246, 0 )

params = ( beta, TFP, gamma, costEXP, costK, sigma, mu )
gridparams  = ( gridLand, gridShocks, initialState, tol )

state_variable_data_labels = []
state_variable_data = []
count = 1
with open( 'statevarvalues.csv', 'rb' ) as csvfile:
    reader = csv.reader( csvfile )
    for row in reader:
        state_variable_data_labels.append( row[0] )
        data_row = []
        if count < 3:
            for element in row[1:]:
                data_row.append( float( element ) )
        else:
            for element in row[1:]:
                data_row.append( int( element ) )
        state_variable_data.append( data_row )
        count +=1

taxRate	 = 0.0
landTax	 = False
defoTax	 = False
deterministic = False
grandfather = 49.246
policyparams= ( taxRate, landTax, defoTax, grandfather, deterministic )

base_instance	    = defo_single( params, gridparams, policyparams )
base_instance.fittedValueIteration()
base_instance.stateSequence( 6 )
base_policy		    = base_instance.policy
base_valueFunction  = base_instance.newvaluefunction
base_agIncome	    = base_instance.outputPath
base_landPath	    = base_instance.landsequence

base_instance.empiricalDistributionPlot( 10000, 6 , None )
base_mean_landholdings = mean( base_instance.sample )
print base_mean_landholdings

plt.title( 'Agricultural GDP' )
plt.plot( state_variable_data[2], base_agIncome, label = 'Baseline' )
plt.plot( state_variable_data[2], state_variable_data[1], label = 'Historical' )
plt.legend()
plt.show()

plt.title( ' Agricultural landholdings ' )
plt.plot( state_variable_data[2], base_landPath, label = 'Model' )
plt.plot( state_variable_data[2], state_variable_data[0], label = 'Historical' )
plt.legend()
plt.show()
