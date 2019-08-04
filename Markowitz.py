from cvxopt import matrix, solvers

# TODO add efficient frontier plot

class MeanVarOpt:
    """Class that manages the Markowitz mean variance model"""
    def __init__(self, returns, std_deviation, correlation, min_ret):
        self.nsec = len(returns)

        self.G = matrix(0.0, (1, self.nsec))
        for i in range(self.nsec):
            self.G[0,i] = -returns[i]
        self.h = matrix([-float(min_ret)])

        self.A = matrix(1.0, (1, self.nsec))
        self.b = matrix([1.0])

        self.P = matrix(0.0, (self.nsec, self.nsec))
        for i in range(self.nsec):
            for j in range(self.nsec - i):
                k = i + j
                self.P[i,k] = correlation[i,k] * std_deviation[i] * std_deviation[k]
                self.P[k,i] = self.P[i,k]

        self.G_nrows = 1
        self.ncols = self.nsec
        self.A_nrows = 1

    def write(self):
        print("objective P:\n{}".format(self.P))

        print("inequality constraints Gx <= h:")
        for i in range(self.G_nrows):
            cons = "[{}]: ".format(i+1)
            for j in range(self.ncols):
                if self.G[i,j] > 0.0:
                     cons += " + {}<{}>".format(self.G[i,j], j+1)
                elif self.G[i,j] < 0.0:
                     cons += " - {}<{}>".format(-self.G[i,j], j+1)
            cons += " <= {}".format(self.h[i])
            print(cons)

        print("\nequality constraints Ax = b:")
        for i in range(self.A_nrows):
            cons = "[{}]: ".format(i+1)
            for j in range(self.ncols):
                if self.A[i,j] > 0.0:
                    cons += " + {}<{}>".format(self.A[i,j], j+1)
                elif self.A[i,j] < 0.0:
                    cons += " - {}<{}>".format(-self.A[i,j], j+1)
            cons += " = {}".format(self.b[i])
            print(cons)

    def addWeightBounds(self, minweight, maxweight):
        self.G = matrix([self.G, matrix(0.0, (2 * self.nsec, self.ncols))])
        self.h = matrix([self.h, matrix(0.0, (2 * self.nsec, 1))])
        offset = self.G_nrows
        self.G_nrows += 2 * self.nsec

        for i in range(self.nsec):
            self.G[offset + 2 * i, i] = -1.0
            self.h[offset + 2 * i]    = -minweight

            self.G[offset + 2 * i + 1, i] = 1.0
            self.h[offset + 2* i  + 1]    = maxweight

    def addSectorLimit(self, sector, limit):
        self.G = matrix([self.G, matrix(0.0, (1, self.ncols))])
        self.h = matrix([self.h, limit])
        last_row = self.G_nrows
        self.G_nrows += 1

        for j in range(len(sector)):
            self.G[last_row, sector[j]] = 1.0

    def addTurnoverLimit(self, reference_weights, max_change):
        #resize the objective
        self.P = matrix([self.P, matrix(0.0, (2 * self.nsec, self.ncols))])
        self.P = matrix([[self.P], [matrix(0.0, (self.ncols + 2 * self.nsec, 2 * self.nsec))]])

        # add columns to G and A
        self.G = matrix([[self.G], [matrix(0.0, (self.G_nrows,  2 * self.nsec))]])
        self.A = matrix([[self.A], [matrix(0.0, (self.A_nrows,  2 * self.nsec))]])
        oldncols = self.ncols
        self.ncols += 2 * self.nsec

        # add space for new constraints
        nbnewcons = 4 * self.nsec + 1
        self.G = matrix([self.G, matrix(0.0, (nbnewcons, self.ncols))])
        self.h = matrix([self.h, matrix(0.0, (nbnewcons, 1))])
        oldnrow = self.G_nrows
        self.G_nrows += 4 * self.nsec + 1

        last_row = self.G_nrows - 1
        for i in range(self.nsec):
            rowoffset = oldnrow + 4 * i
            yvarid = oldncols + 2 * i
            zvarid = oldncols + 2 * i + 1

            self.G[rowoffset, i] = 1.0
            self.G[rowoffset, yvarid] = -1.0
            self.h[rowoffset] = reference_weights[i]

            self.G[rowoffset + 1, i] = -1.0
            self.G[rowoffset + 1, zvarid] = -1.0
            self.h[rowoffset + 1] = -reference_weights[i]

            self.G[rowoffset + 2, yvarid] = -1.0

            self.G[rowoffset + 3, zvarid] = -1.0

            self.G[last_row, yvarid] = 1.0
            self.G[last_row, zvarid] = 1.0

        self.h[last_row] = max_change
            
    def addTransactionCosts(self, reference_weights, sell_cost, buy_cost):
        #resize the objective
        self.P = matrix([self.P, matrix(0.0, (2 * self.nsec, self.ncols))])
        self.P = matrix([[self.P], [matrix(0.0, (self.ncols + 2 * self.nsec, 2 * self.nsec))]])

        # add columns to G and A
        self.G = matrix([[self.G], [matrix(0.0, (self.G_nrows,  2 * self.nsec))]])
        self.A = matrix([[self.A], [matrix(0.0, (self.A_nrows,  2 * self.nsec))]])
        oldncols = self.ncols
        self.ncols += 2 * self.nsec

        # add space for new constraints
        nbnewcons = 4 * self.nsec
        self.G = matrix([self.G, matrix(0.0, (nbnewcons, self.ncols))])
        self.h = matrix([self.h, matrix(0.0, (nbnewcons, 1))])
        oldnrow = self.G_nrows
        self.G_nrows += 4 * self.nsec

        last_row = self.G_nrows - 1
        for i in range(self.nsec):
            rowoffset = oldnrow + 4 * i
            yvarid = oldncols + 2 * i
            zvarid = oldncols + 2 * i + 1

            self.G[0,yvarid] = sell_cost[i]
            self.G[0,zvarid] = buy_cost[i]

            self.G[rowoffset, i] = 1.0
            self.G[rowoffset, yvarid] = -1.0
            self.h[rowoffset] = reference_weights[i]

            self.G[rowoffset + 1, i] = -1.0
            self.G[rowoffset + 1, zvarid] = -1.0
            self.h[rowoffset + 1] = -reference_weights[i]

            self.G[rowoffset + 2, yvarid] = -1.0

            self.G[rowoffset + 3, zvarid] = -1.0

    def solve(self):
        sol = solvers.qp(self.P, matrix(0.0, (self.ncols, 1)), self.G, self.h, self.A, self.b)
        return sol['x'][:3]

if __name__ == "__main__":
    returns  = matrix([0.1073, 0.0737, 0.0627])
    std_dev = matrix([0.1667, 0.1055, 0.034])
    correlation = matrix([[1.0, 0.2199, 0.0366], [0.2199, 1, -0.0545], [0.0366, -0.0545, 1.0]])

    x = MeanVarOpt(returns, std_dev, correlation, 0.07)
    x.addWeightBounds(0.1, 0.45)
    #x.addSectorLimit([2], 0.5)
    x.addTurnoverLimit([0.33, 0.33, 0.33], 0.3)
    x.addTransactionCosts([0.33, 0.33, 0.33], [0.02, 0.01, 0.001], [0.02, 0.01, 0.001])
    x.write()
    a = x.solve()
    print(a)
