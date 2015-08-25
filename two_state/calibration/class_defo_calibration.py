"""
A single sector model to investigate various policies to reduce deforestation
"""

from ecdf import ECDF
from numpy import linspace, array, nan_to_num, reshape, meshgrid, std
from numpy.random import uniform, lognormal
from scipy import mean, interpolate
from scipy.optimize import fminbound
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import cm


class defo_calibration:
    def __init__(
            self,
            (beta, TFP, gamma, theta, costEXP, costTFP, mu,
                sigma, landtoemissions, paymentstodollars),
            (gridLand, gridPayments, initialState, tol),
            ):
        # Initialises code with parameters
        self.beta = beta            # discount rate
        self.TFP = TFP              # TFP on ag output
        self.gamma = gamma          # output share land
        self.theta = theta          # utility parameter
        self.costEXP = costEXP      # exponent on cost function
        self.costTFP = costTFP      # multiplicative constant on cost fn
        self.mu = mu                # mean of defo error
        self.sigma = sigma          # standard deviation of defo error
        # parameter conversion factors
        self.landtoemissions = landtoemissions
        self.paymentstodollars = paymentstodollars
        # constructing grid
        self.shocks = lognormal(mu, sigma, 50)
        self.grid1 = gridLand
        self.grid2 = gridPayments
        grid = meshgrid(self.grid2, self.grid1)
        grid[0], grid[1] = grid[1], grid[0]
        self.gridLand = grid[0]
        self.gridPayments = grid[1]
        self.state = array(grid).reshape(2, -1).T
        self.maxLand = max(gridLand)    # maximum amount of landholdings
        self.minLand = min(gridLand)    # minimum amount of landholdings
        self.initialState = initialState  # initial state value for sequences
        self.tol = tol                  # tolerance for fitted value iteration
        ##

    def maximiser(self, h, a, b):
        return float(fminbound(lambda x: -h(x), a, b))

    def productionFunction(self, land):
        return self.TFP * land**self.gamma

    def costFunction(self, land, nextLand):
        return (self.costTFP / 2.0) * (nextLand-land)**self.costEXP

    def utility(self, consumption):
        return consumption

    def bellman(self):
        """
        This is the approximate Bellman operator
        Parameters: w is a vectorised function
        Returns: An interpolated instance
        """
        vals = []
        policy = []
        count = 0
        for s in self.state:
            land = s[0]
            costshock = s[1]
            h = lambda (nextLand): self.utility(
                self.productionFunction(land))-(1/costshock) *\
                self.costFunction(land, nextLand)+self.beta *\
                mean(self.valuefunction(nextLand, self.shocks))
            x = self.maximiser(h, self.minLand, self.maxLand)
            vals.append(h(x))
            policy.append(x)
            count += 1
        vals = array(nan_to_num(vals)).squeeze()
        policy = array(nan_to_num(policy)).squeeze()
        self.newvaluefunction = interpolate.LinearNDInterpolator(
            self.state, vals)
        self.newpolicy = interpolate.LinearNDInterpolator(self.state, policy)

    def fittedValueIteration(self, initialV=None, initialP=None):
        "Initialising value and policy functions"
        if initialV is None:
            self.valuefunction = self.productionFunction(
                self.gridLand.ravel())+(1/self.gridPayments.ravel())
            self.valuefunction = interpolate.LinearNDInterpolator(
                self.state, self.valuefunction)
        else:
            self.valuefunction = initialV
        if initialP is None:
            p = self.gridLand.ravel()
            self.policy = interpolate.LinearNDInterpolator(self.state, p)
        else:
            self.policy = initialP
            print 'Approximating value function'
            print 'No. Iterations   || vNew - v ||_sup'
        count = 0
        while 1:
            self.bellman()
            error = abs(self.newvaluefunction(self.state) -
                        self.valuefunction(self.state))
            err = max(error)
            print count, '      ', round(err, 5)
            if err < self.tol:
                return self.newvaluefunction, self.newpolicy
            self.valuefunction = self.newvaluefunction
            self.policy = self.newpolicy
            count += 1

    def transferFinder(self, valueBaseline):
        # takes a baseline policy function, a policy function with a policy
        # and an initial state.
        # finds the period zero transfer required equalise welfare
        initialtransfer = valueBaseline(self.initialState) - \
            self.newvaluefunction(self.initialState)
        if abs(initialtransfer) < self.tol:
            self.transfer = 0
        else:
            self.transfer = initialtransfer

    def landSequenceGenerator(self, sequenceLength, baselinePolicy):
        """
        takes an optimal policy and an initial pair.
        returns a path of landholdings
        """
        landsequence = [self.initialState[0]]
        shocksequence = lognormal(self.mu, self.sigma, sequenceLength)
        for t in range(sequenceLength):
            landsequence.append(
                self.newpolicy(landsequence[-1], shocksequence[t]))
        self.landsequence = array(landsequence)
        self.emissionssequence = self.landtoemissions * array(landsequence)

    def sampleTaker(self, sampleLength, sequenceLength, baselinePolicy):
        """
        takes a sequence of paths
        returns a sequence with the time-th element in each one.
        """
        output_land = []
        output_defo_sd = []
        for t in range(sampleLength):
            self.landSequenceGenerator(sequenceLength, baselinePolicy)
            output_land.append(self.landsequence[-1])
            deforestation = self.landsequence[1:] - self.landsequence[:-1]
            output_defo_sd.append(std(deforestation))
        self.sample_land = array(output_land)
        self.sample_d_sd = array(output_defo_sd)

    def empiricalDistributionPlot(self, sample, bounds=None):
        if bounds:
            lower = bounds[0]
            upper = bounds[1]
        else:
            lower = min(sample)
            upper = max(sample)
        plotDomain = linspace(lower, upper, len(sample))
        empiricalCDF = ECDF([uniform(0, 1) for i in range(len(sample))])
        empiricalCDF.observations = sample
        obs = []
        for j in range(len(sample)):
            obs.append(empiricalCDF(plotDomain[j]))
        ecdf_sample = array(obs)
        plt.plot(plotDomain, ecdf_sample)
        plt.show()

    def threeDimPlot(self, function):
        X = self.gridLand
        Y = self.gridPayments
        function = function(self.state)
        function = reshape(function, (len(self.grid1), len(self.grid2)))
        fig1 = plt.figure(1)
        ax = p3.Axes3D(fig1)
        ax.plot_surface(
            X, Y, function, rstride=1, cstride=1, cmap=cm.jet,
            linewidth=0, antialiased=False)
        ax.set_xlabel('Land')
        ax.set_ylabel('Shocks')
        ax.set_zlabel('Deforestation')
        plt.show()
