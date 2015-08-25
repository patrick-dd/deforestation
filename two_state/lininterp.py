from scipy import interp

class LinInterp:
    "Provides linear interpolation in one dimension"

    def __init__( self, X, Y ):
        self.X, self.Y = X, Y

    def __call__( self, Z ):
        return interp(Z, self.X, self.Y)
