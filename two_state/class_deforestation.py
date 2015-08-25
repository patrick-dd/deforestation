"""
A single sector model to investigate various policies to reduce deforestation
"""

from ecdf import ECDF
from numpy import linspace, array, nan_to_num, reshape, meshgrid, nansum
from numpy.random import uniform, lognormal
from scipy import exp, mean, interpolate
from scipy.optimize import fminbound
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import cm


class class_defo:
    def __init__(
            self,
            (beta, TFP, gamma, theta, costEXP, costTFP, mu, sigma,
                landtoemissions, paymentstodollars),
            (gridLand, gridShocks, initialState, tol),
            (policyRate, adPolicyFlag, threshold)):
        # initialises code with parameters
        self.beta = beta                # discount rate
        self.TFP = TFP                  # TFP on ag output
        self.gamma = gamma              # output share land
        self.theta = theta              # initial intertemp elas subs
        self.costEXP = costEXP          # exponent on cost function
        self.costTFP = costTFP          # constant on cost function
        self.mu = mu                    # mean of defo cost shock
        self.sigma = sigma              # std dev of defo cost shock

        # parameter conversion factors
        self.landtoemissions = landtoemissions
        self.paymentstodollars = paymentstodollars

        # constructing grid
        self.shocks = lognormal(mu, sigma, 1000)
        self.grid1 = gridLand
        self.grid2 = gridShocks
        grid = meshgrid(self.grid2, self.grid1)
        grid[0], grid[1] = grid[1], grid[0]
        self.gridLand = grid[0]
        self.gridShocks = grid[1]
        self.state = array(grid).reshape(2, -1).T
        self.maxLand = max(gridLand)  # maximum amount of landholdings
        self.minLand = min(gridLand)  # minimum amount of landholdings
        self.initialState = initialState    # initial state for time iteration
        self.tol = tol           # tolerance for fitted value iteration

        # avoided deforestation policy parameters
        # flag for policy type 'defoTax', 'landTax', 'adSub'
        self.policyRate = policyRate        # tax rate
        self.adPolicyFlag = adPolicyFlag    # flag for policy type
        self.threshold = threshold          # threshold for policy

    def maximiser(self, h, a, b):
        """
        A wrapper for the scipy minimize function
        Takes:
            h - function
            a - lower bound for argument (float)
            b - upper bound for argument (float)
        Returns:
            maximiser (float)
        """
        return float(fminbound(lambda x: -h(x), a, b))

    def productionFunction(self, land):
        """
        Aggregate agriculture output function
        Takes:
            land - levels of landholdings (float)
        Returns:
            levels of output value (float)
        """
        return self.TFP * land**self.gamma

    def costFunction(self, land, nextLand):
        """
        Deforestation effort cost function
        Takes:
            land - current landholdings (float)
            nextLand - next period's landholdings (float)
        Returns:
            effort costs (float)
        """
        return (self.costTFP / 2.0) * (nextLand-land)**self.costEXP

    def utility(self, consumption):
        """
        Utility of consumption
        Takes:
            consumption - levels (float)
        Returns
            utility levels (float)
        """
        return 1-exp(-self.theta*consumption)

    def adPolicy(self, land, nextLand):
        """
        Calculates net payments to agent from deforestation with policy
        Given policy flags (str):
            landTax - tax on agricultural landholdings levels
            defoTax - tax on deforestation levels = changes in agricultural
            landholdings
            adSub - subsidy for avoided deforestation
        and tax rate policyRate (float).
        Takes:
            land - current landholdings (float)
            nextLand - next period's landholdings (float)
        Returns:
            Tax payments of avoided deforestation subsidies (float)
        """
        if self.adPolicyFlag == 'landTax':
            return - self.policyRate * (land-self.threshold)
        elif self.adPolicyFlag == 'defoTax':
            return - self.policyRate * (nextLand-land)
        elif self.adPolicyFlag == 'adSub':
            return self.policyRate * max(0, self.threshold+land-nextLand)
        else:
            return 0

    def bellman(self):
        """
        This is the approximate Bellman operator
        Takes:
            old value function w (interpolated function)
        Returns:
            new value function (interpolated instance)
        """
        vals = []
        policy = []
        count = 0
        for s in self.state:
            land = s[0]
            costshock = s[1]
            h = lambda (nextLand):\
                self.utility(self.productionFunction(land))\
                + self.adPolicy(land, nextLand)\
                - (1/costshock) * self.costFunction(land, nextLand)\
                + self.beta * mean(self.valuefunction(nextLand, self.shocks))
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
        """
        Algorithm for numerical fitted value iteration
        Takes:
            initialV - initial function         (function)
            initialP - initial policy function  (function)
        Returns:
            approximate value function          (function)
            approximate optimal policy function (function)
        Initialising value and policy functions
        """
        if initialV is None:
            self.valuefunction =\
                self.productionFunction(self.gridLand.ravel())\
                + (1/self.gridShocks.ravel())
            self.valuefunction =\
                interpolate.LinearNDInterpolator(
                    self.state, self.valuefunction)
        else:
            self.valuefunction = initialV
        if initialP is None:
            p = self.gridLand.ravel()
            self.policy = interpolate.LinearNDInterpolator(self.state, p)
        else:
            self.policy = initialP
            print 'Approximating value function'
            print 'No. Iterations  || vNew-v ||_sup'
        count = 0
        while 1:
            self.bellman()
            error = abs(
                self.newvaluefunction(self.state)
                - self.valuefunction(self.state))
            err = max(error)
            print count, '      ', round(err, 5)
            if err < self.tol:
                return self.newvaluefunction, self.newpolicy
            self.valuefunction = self.newvaluefunction
            self.policy = self.newpolicy
            count += 1

    def transferFinder(self, valueBaseline):
        """
        Calculates the compensating variation for accepting policy
        Takes:
            valueBaseline - a baseline value function (function)
        Returns:
            period zero transfer required equalise welfare (float)
        """
        initialtransfer = \
            valueBaseline(self.initialState) \
            - self.newvaluefunction(self.initialState)
        if abs(initialtransfer) < self.tol:
            self.transfer = 0
        else:
            self.transfer = initialtransfer

    def landSequenceGenerator(self, sequenceLength, baselinePolicy):
        """
        takes an optimal policy and an initial pair. returns a path of
        landholdings
        """
        landsequence = [self.initialState[0]]
        landsequence_base = [self.initialState[0]]
        paymentsequence = []
        shocksequence = lognormal(self.mu, self.sigma, sequenceLength)
        if self.adPolicyFlag is None:
            for t in range(sequenceLength):
                landsequence.append(
                    self.newpolicy(landsequence[-1], shocksequence[t]))
        else:
            for t in range(sequenceLength):
                landsequence.append(
                    self.newpolicy(landsequence[-1], shocksequence[t]))
                landsequence_base.append(
                    baselinePolicy(landsequence_base[-1], shocksequence[t]))
                paymentsequence.append(
                    self.adPolicy(landsequence[-2], landsequence[-1]))
        self.landsequence = array(landsequence)
        self.landsequence_base = array(landsequence_base)
        self.emissionssequence = self.landtoemissions * array(landsequence)
        self.emissionssequence_base = \
            self.landtoemissions * \
            array(landsequence_base)
        self.paymentsequence = self.paymentstodollars * array(
            [b * s for (b, s) in zip(self.betaseq, array(paymentsequence))])

    def sampleTaker(self, sampleLength, sequenceLength, baselinePolicy):
        """
        takes a sequence of paths
        returns a sequence with the time-th element in each one.
        """
        output_land = []
        output_avoidedDeforestation = []
        output_payment = []
        output_costperco2 = []
        output_netcostperco2 = []
        output_disc_changeProduction = []
        self.betaseq = array([self.beta**t for t in range(sequenceLength)])
        transferpath = self.paymentstodollars * array(
            [b * self.transfer * (1-self.beta) for b in self.betaseq])
        for t in range(sampleLength):
            # collecting sequence
            self.landSequenceGenerator(sequenceLength, baselinePolicy)
            # avoided deforestation
            if (self.landsequence_base[-1]-self.landsequence[-1]) < self.tol:
                avoideddeforestation = 0
            else:
                avoideddeforestation = \
                    self.emissionssequence_base[-1] - \
                    self.emissionssequence[-1]
            output_avoidedDeforestation.append(avoideddeforestation)
            output_land.append(self.landsequence[-1])
            # discounted sum payments for avoided deforestation
            if abs(sum(self.paymentsequence)) < self.tol:
                output_payment.append(0)
            else:
                output_payment.append(nansum(self.paymentsequence))
            # cost effectiveness
            if output_avoidedDeforestation[-1] == 0:
                output_costperco2.append(0)
            else:
                if self.adPolicyFlag == 'adSub':
                    output_costperco2.append(
                        output_payment[-1] / output_avoidedDeforestation[-1])
                else:
                    output_costperco2.append(
                        nansum(transferpath) /
                        output_avoidedDeforestation[-1])
            # net cost
            if output_avoidedDeforestation[-1] == 0:
                output_netcostperco2.append(0)
            else:
                if self.adPolicyFlag == 'adSub':
                    output_netcostperco2.append(
                        (output_payment[-1] - nansum(transferpath)) /
                        output_avoidedDeforestation[-1])
                else:
                    output_netcostperco2.append(
                        (output_payment[-1] + nansum(transferpath))
                        / output_avoidedDeforestation[-1])
            # difference in output
            changeOutput = self.paymentstodollars * array(
                [b * (o - o_b) for (b, o, o_b) in
                    zip(
                        self.betaseq,
                        [self.productionFunction(l) for l in
                            self.landsequence],
                        [self.productionFunction(l) for l in
                            self.landsequence_base]
                    )])
            if abs(sum(changeOutput)) < self.tol:
                output_disc_changeProduction.append(0)
            else:
                output_disc_changeProduction.append(sum(changeOutput))
        self.sample_avoidedDeforestation = array(output_avoidedDeforestation)
        self.sample_payments = array(output_payment)
        self.sample_costperco2 = array(output_costperco2)
        self.sample_netcostperco2 = array(output_netcostperco2)
        self.sample_discChangeProduction = \
            array(output_disc_changeProduction)

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
        Y = self.gridShocks
        function = function(self.state)
        function = reshape(function, (len(self.grid1), len(self.grid2)))
        fig1 = plt.figure(1)
        ax = p3.Axes3D(fig1)
        ax.plot_surface(
            X, Y, function, rstride=1, cstride=1,
            cmap=cm.jet, linewidth=0, antialiased=False)
        ax.set_xlabel('Land')
        ax.set_ylabel('Shocks')
        ax.set_zlabel('Deforestation')
        plt.show()
