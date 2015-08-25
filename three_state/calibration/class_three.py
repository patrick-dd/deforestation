"""
A dual sector stochastic model to investigate
various policies to reduce deforestation
"""

from ecdf import ECDF
from numpy import linspace, array, nan_to_num, reshape, meshgrid, nansum, std
from numpy.random import uniform
from scipy import exp, mean, interpolate
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import cm


class deforestation_growth:

    def __init__(
            self,
            (beta, TFP, theta, alpha, gamma, costEXP, costTFP, delta),
            (mu_c, sigma_c, dist_c, mu_l, sigma_l, dist_l, mu_y, sigma_y,
                dist_y),
            (gridLand, gridShock, gridIncome, initialState, tol,
                landtoemissions, paymentstodollars),
            (policyRate, adPolicyFlag, threshold)
            ):
        """
        Initialises code with parameters and functions
        """
        "Preference and technology parameters"
        self.beta = beta                # discount rate
        self.theta = theta                # utility parameter
        self.TFP = TFP                    # TFP
        self.alpha = alpha                # output share capital
        self.gamma = gamma                 # output share land
        self.costEXP = costEXP            # exponent on cost function
        self.costTFP = costTFP            # constant on cost function
        self.delta = delta                # capital depreciation rate
        "Shock parameters"
        self.mu_c = mu_c                # mean cost shock
        self.sigma_c = sigma_c            # std dev cost shock
        self.dist_c = dist_c            # distribtion cost shock
        self.mu_l = mu_l                # mean land shock
        self.sigma_l = sigma_l            # std dev land shock
        self.dist_l = dist_l            # distribution land shock
        self.mu_y = mu_y                # mean output shock
        self.sigma_y = sigma_y            # std dev output shock
        self.dist_y = dist_y            # distribution output shock
        "Conversion parameters"
        self.landtoemissions = landtoemissions            # land in CO2e
        self.paymentstodollars = paymentstodollars        # payments in USD2010
        "Grid parameters"
        self.costshocksequence = self.dist_c(self.mu_c, self.sigma_c, 30)
        self.grid1 = gridIncome
        self.grid2 = gridLand
        self.grid3 = gridShock
        grid = meshgrid(self.grid2, self.grid1, self.grid3)
        grid[0], grid[1] = grid[1], grid[0]
        self.gridIncome = grid[0]
        self.gridLand = grid[1]
        self.gridShock = grid[2]
        self.state = array(grid).reshape(3, -1).T        # state
        self.maxLand = max(gridLand)
        self.minLand = min(gridLand)
        self.initialState = initialState
        self.tol = tol                            # tolerance for fvi
        "Policy parameters"
        self.policyRate = policyRate            # tax level
        self.adPolicyFlag = adPolicyFlag        # policy type
        self.threshold = threshold              # reference level

    def nDmaximum(self, h, guess, bound):
        "wrapper for function minimiser"
        g = lambda x: -h(x)
        x = fmin_l_bfgs_b(
            g, guess, bounds=bound, approx_grad=True, disp=False
        )
        return float(h(x[0])), x[0]

    def productionFunction(self, capital, land):
        return self.TFP * (capital**self.alpha) * (land**self.gamma)

    def costFunction(self, land, costshock, nextLand):
        "Deforestation effort cost function"
        return (1 / costshock) * (self.costTFP / 2.0) * \
            (nextLand - land)**self.costEXP

    def utility(self, consumption):
        return 1 - exp(- self.theta * consumption)

    def adPolicy(self, land, nextLand):
        """
        Calculates net payments to agent from deforestation with policy
        adPolicy is an indicator function for the relevant policy
        taxedLand is the area of land that is taxed
        """
        if self.adPolicyFlag == 'landTax':
            return - self.policyRate * (nextLand - self.threshold)
        elif self.adPolicyFlag == 'defoTax':
            return - self.policyRate * (nextLand - land)
        elif self.adPolicyFlag == 'adSub':
            return self.policyRate * max(0, self.threshold + land - nextLand)
        else:
            return 0

    def bellman(self):
        """
        This is the approximate Bellman operator
        Parameters: w is a vectorised function
        Returns: An interpolated instance
        """
        vals = []
        policyK = []
        policyL = []
        count = 0
        for s in self.state:
            income = s[0]
            land = s[1]
            costshock = s[2]
            h = lambda (investment, nextLand): self.utility(
                income - investment) - \
                self.costFunction(land, costshock, nextLand) + \
                self.beta * mean(self.valuefunction(
                    self.productionFunction(investment, nextLand)
                    + (1-self.delta) * investment +
                    self.adPolicy(land, nextLand),
                    nextLand, self.costshocksequence))
            v, x = self.nDmaximum(
                h, (self.policy[0](income, land, costshock),
                    self.policy[1](income, land, costshock)),
                ((0, income), (land, self.maxLand))
                )
            vals.append(v)
            policyK.append(x[0])
            policyL.append(x[1])
            count += 1
        vals = array(nan_to_num(vals)).squeeze()
        policyK = array(nan_to_num(policyK)).squeeze()
        policyL = array(nan_to_num(policyL)).squeeze()
        self.newvaluefunction = \
            interpolate.LinearNDInterpolator(self.state, vals)
        self.newpolicy = (
            interpolate.LinearNDInterpolator(self.state, policyK),
            interpolate.LinearNDInterpolator(self.state, policyL)
        )

    def fittedValueIteration(self, initialV=None, initialP=None):
        "Initialising value and policy functions"
        if initialV is None:
            self.valuefunction = self.productionFunction(
                self.gridIncome.ravel()/1.5, self.gridLand.ravel()) \
                + self.gridShock.ravel()
            self.valuefunction = interpolate.LinearNDInterpolator(
                self.state, self.valuefunction)
        else:
            self.valuefunction = initialV
        if initialP is None:
            pK = self.gridIncome.ravel()*0.66
            pL = self.gridLand.ravel()*1.01
            self.policy = (
                interpolate.LinearNDInterpolator(self.state, pK),
                interpolate.LinearNDInterpolator(self.state, pL)
            )
        else:
            self.policy = initialP
        count = 0
        while 1:
            self.bellman()
            error = abs(self.newvaluefunction(self.state) -
                        self.valuefunction(self.state))
            err = max(error)
            print count, '        ', round(err, 5)
            if err < self.tol:
                return self.newvaluefunction, self.newpolicy
            self.valuefunction = self.newvaluefunction
            self.policy = self.newpolicy
            count += 1

    def transferFinder(self, valueBaseline):
        """
        Takes a baseline policy function, a policy function with a policy and
        an initial state
        Finds the period zero transfer required equalise welfare
        """
        initialtransfer = valueBaseline(self.initialState) - \
            self.newvaluefunction(self.initialState)
        if abs(initialtransfer) < self.tol:
            self.transfer = 0
        else:
            self.transfer = initialtransfer

    def stateSequenceGenerator(self, sequenceLength, baselinePolicy):
        """
        Returns paths of income, landholdings and payments
        """
        incomesequence = [self.initialState[0]]
        incomesequence_base = [self.initialState[0]]
        landsequence = [self.initialState[1]]
        landsequence_base = [self.initialState[1]]
        paymentsequence = []
        costshocksequence = self.dist_c(self.mu_c, self.sigma_c,
                                        sequenceLength)
        if self.adPolicyFlag is None:
            for t in range(sequenceLength):
                investment = self.newpolicy[0](
                    incomesequence[-1], landsequence[-1], costshocksequence[t]
                    )
                nextLand = self.newpolicy[1](
                    incomesequence[-1], landsequence[-1], costshocksequence[t]
                    )
                output = self.productionFunction(investment, nextLand) + \
                    (1-self.delta) * investment
                payments = self.adPolicy(nextLand, landsequence[-1])
                incomesequence.append(output)
                landsequence.append(nextLand)
        else:
            for t in range(sequenceLength):
                investment = self.newpolicy[0](
                    incomesequence[-1], landsequence[-1], costshocksequence[t]
                    )
                nextLand = self.newpolicy[1](
                    incomesequence[-1], landsequence[-1], costshocksequence[t]
                    )
                output = self.productionFunction(investment, nextLand) + \
                    (1-self.delta)*investment
                payments = self.adPolicy(nextLand, landsequence[-1])
                incomesequence.append(output + payments)
                landsequence.append(nextLand)
                paymentsequence.append(payments)
                "Baseline policies with same initial state and shocks"
                investment = baselinePolicy[0](
                    incomesequence_base[-1], landsequence_base[-1],
                    costshocksequence[t]
                    )
                nextLand = baselinePolicy[1](
                    incomesequence_base[-1], landsequence_base[-1],
                    costshocksequence[t]
                    )
                output = self.productionFunction(investment, nextLand) + \
                    (1-self.delta)*investment
                incomesequence.append(output)
                landsequence.append(nextLand)
        self.landsequence = array(landsequence)
        self.landsequence_base = array(landsequence_base)
        self.incomesequence = array(incomesequence)
        self.incomesequence_base = array(incomesequence_base)
        self.emissionssequence = self.landtoemissions * \
            array(landsequence)
        self.emissionssequence_base = self.landtoemissions * \
            array(landsequence_base)
        self.paymentsequence = \
            self.paymentstodollars * array(
                [b*s for (b, s) in zip(self.betaseq, array(paymentsequence))]
            )

    def sampleTaker(self, sampleLength, sequenceLength, baselinePolicy):
        """
        Sample length (int) is the length of the policy
        Sequence length is the (int) is the length of the sample
        Returns a sample of statistics
        nansum is used instead of sum because sometimes the shock moves the
        state outside the grid
        """
        output_land = []
        output_income = []
        output_diff_income = []
        output_income_base = []
        output_avoidedDeforestation = []
        output_payment = []
        output_costperco2 = []
        output_netcostperco2 = []
        output_defo_sd = []
        "npv of transfer"
        self.betaseq = array([self.beta**t for t in range(sequenceLength)])
        transferpath = \
            self.paymentstodollars * array(
                [b * self.transfer * (1-self.beta) for b in self.betaseq]
            )
        for t in range(sampleLength):
            self.stateSequenceGenerator(sequenceLength, baselinePolicy)
            if (
                (self.landsequence_base[-1] - self.landsequence[-1])
                < self.tol
            ):
                avoideddeforestation = 0
            else:
                avoideddeforestation = \
                    self.emissionssequence_base[-1] - \
                    self.emissionssequence[-1]
            output_avoidedDeforestation.append(avoideddeforestation)
            output_land.append(self.landsequence[-1])
            deforestation = \
                self.landsequence[1:] - self.landsequence[:-1]
            output_defo_sd.append(std(deforestation))
            output_income.append(self.incomesequence[-1])
            output_income_base.append(self.incomesequence_base[-1])
            "discounted sum payments for avoided deforestation"
            if abs(sum(self.paymentsequence)) < self.tol:
                output_payment.append(0)
            else:
                output_payment.append(nansum(self.paymentsequence))
            "cost effectiveness"
            if output_avoidedDeforestation[-1] == 0:
                output_costperco2.append(0)
            else:
                if self.adPolicyFlag == 'adSub':
                    output_costperco2.append(
                        output_payment[-1] / output_avoidedDeforestation[-1]
                    )
                else:
                    output_costperco2.append(
                        nansum(transferpath) / output_avoidedDeforestation[-1]
                    )
            "cost effectiveness net of transfer"
            if output_avoidedDeforestation[-1] == 0:
                output_netcostperco2.append(0)
            else:
                if self.adPolicyFlag == 'adSub':
                    output_netcostperco2.append(
                        (output_payment[-1] + sum(transferpath)) /
                        output_avoidedDeforestation[-1]
                    )
                else:
                    output_netcostperco2.append(
                        (- output_payment[-1] + nansum(transferpath)) /
                        output_avoidedDeforestation[-1]
                    )
            "difference in income"
            diff_income = self.paymentstodollars * array(
                [b * (y-y_b) for (b, y, y_b) in zip(
                    self.betaseq, output_income, output_income_base
                )])
            if abs(sum(diff_income)) < self.tol:
                output_diff_income.append(0)
            else:
                output_diff_income.append(sum(diff_income))
        self.sample_income = array(output_income)
        self.sample_land = array(output_land)
        self.sample_avoidedDeforestation = array(output_avoidedDeforestation)
        self.sample_payments = array(output_payment)
        self.sample_costperco2 = array(output_costperco2)
        self.sample_netcostperco2 = array(output_netcostperco2)
        self.sample_diff_income = array(output_diff_income)
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
        Y = self.gridShock
        function = function(self.state)
        function = reshape(function, (len(self.grid1), len(self.grid2)))
        fig1 = plt.figure(1)
        ax = p3.Axes3D(fig1)
        ax.plot_surface(
            X, Y, function, rstride=1, cstride=1, cmap=cm.jet,
            linewidth=0, antialiased=False
        )
        ax.set_xlabel('Land')
        ax.set_ylabel('Shocks')
        ax.set_zlabel('Deforestation')
        plt.show()
