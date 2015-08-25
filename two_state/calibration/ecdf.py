class ECDF:
    """
    Creates an empirical distribution function
    """
    def __init__(self, observations):
        # observations is a sequence (X_t), t in [1,n]
        self.observations = observations

    def __call__(self, x):
        # creates the function
        counter = 0.0
        for obs in self.observations:
            if obs <= x:
                counter += 1
        return counter/len(self.observations)
