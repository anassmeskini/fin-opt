from cvxopt import matrix, solvers

class MinL1RiskModel:
    def __init__(self, historical_returns, min_return, min_weight, max_weight):
        (self.nsec, self.T) = historical_returns.size

        self.ncols = self.nsec + 2 * self.T
        self.A_nrows = self.T + 1
        self.A = matrix(0.0, (self.A_nrows, self.ncols))
        self.b = matrix(0.0, (self.A_nrows, 1))

        self.G_nrows = 2*(self.T + self.nsec) + 1
        self.G = matrix(0.0, (self.G_nrows, self.ncols))
        self.h = matrix(0.0, (self.G_nrows, 1))

        self.c = matrix(0.0, (self.ncols, 1))

        #compute mean return
        u = matrix(0.0, (self.nsec, 1))
        for t in range(self.T):
            for i in range(self.nsec):
                u[i] += historical_returns[i,t]

        u /= self.T

        self.h[0] = -min_return
        for i in range(self.nsec):
            # sum w * ret >= min_ret
            self.G[0,i] = -u[i]

            # w_i <= max_weight
            self.G[1 + 2 * i, i] = 1.0
            self.h[1 + 2 * i] = max_weight

            # w_i >= min_weight
            self.G[2 * (i + 1), i] = -1.0
            self.h[2 * (i + 1)] = -min_weight

            # sum w_i = 1
            self.A[0,i] = 1.0

        self.b[0] = 1.0

        G_offset = 1 + 2 * self.nsec
        A_offset = 1

        for t in range(self.T):
            yvarid = self.nsec + 2 * t
            zvarid = self.nsec + 2 * t + 1

            self.A[A_offset, yvarid] = 1.0
            self.A[A_offset, zvarid] = -1.0
            for i in range(self.nsec):
                self.A[A_offset, i] = u[i] - historical_returns[i, t]
            self.b[A_offset] = 0.0
            
            A_offset += 1

            self.G[G_offset, yvarid] = -1.0
            self.G[G_offset + 1, zvarid] = -1.0

            G_offset += 2

            self.c[yvarid] = 1.0
            self.c[zvarid] = 1.0

    def write(self):
        print("objective:\n{}".format(self.c))
        
        print("inequalities:")
        for i in range(self.G.size[0]):
            cons = "[{}]: ".format(i+1)
            for j in range(self.ncols):
                if self.G[i,j] > 0.0:
                     cons += " + {}<{}>".format(self.G[i,j], j+1)
                elif self.G[i,j] < 0.0:
                     cons += " - {}<{}>".format(-self.G[i,j], j+1)
            cons += " <= {}".format(self.h[i])
            print(cons)

        print("equalities:")
        for i in range(self.A.size[0]):
            cons = "[{}]: ".format(i+1)
            for j in range(self.ncols):
                if self.A[i,j] > 0.0:
                    cons += " + {}<{}>".format(self.A[i,j], j+1)
                elif self.A[i,j] < 0.0:
                    cons += " - {}<{}>".format(-self.A[i,j], j+1)
            cons += " = {}".format(self.b[i])
            print(cons)

    def solve(self):
        #solvers.options['show_progress'] = False
        sol = solvers.lp(self.c, self.G, self.h, self.A, self.b)
        return sol['x'][:self.nsec]

    def addSectorLimit(self, sector, limit):
        self.G = matrix([self.G, matrix(0.0, (1, self.ncols))])
        self.h = matrix([self.h, matrix([limit])])
        lastrowid = self.G.size[0] - 1
        for i in range(len(sector)):
            self.G[lastrowid, sector[i]] = 1.0

    def addTurnoverLimit(self, reference_weights, max_change):
        newcols = 2 * self.nsec
        newGrows = 2 * self.nsec + 1
        Grows = self.G.size[0]
        Arows = self.A.size[0]
        oldncols = self.ncols

        assert oldncols == self.G.size[1]

        # add new columns
        self.G = matrix([[self.G], [matrix(0.0, (Grows, newcols))]])

        self.A = matrix([[self.A], [matrix(0.0, (Arows, newcols))]])

        self.c = matrix([self.c, matrix(0.0, (newcols, 1))])

        self.ncols += newcols
        # add new rows to G
        offset = self.G.size[0]
        self.G = matrix([self.G, matrix(0.0, (2 * self.nsec + 1, self.ncols))])
        self.h = matrix([self.h, matrix(0.0, (newGrows, 1))])

        lastrow = self.G.size[0] - 1
        for i in range(self.nsec):
            vvarid = oldncols + 2 * i
            wvarid = oldncols + 2 * i + 1

            self.G[offset, i] = 1.0
            self.G[offset, vvarid] = -1.0
            self.h[offset] = reference_weights[i]

            self.G[offset + 1, i] = -1.0
            self.G[offset + 1, wvarid] = -1.0
            self.h[offset + 1] = -reference_weights[i]

            offset += 2

            self.G[lastrow, vvarid] = 1.0
            self.G[lastrow, wvarid] = 1.0

        self.h[lastrow] = max_change

if __name__ == "__main__":
    historical_returns  = matrix([[0.1073, 0.0737, 0.0627], [0.085, 0.053, 0.077], [0.2, 0.08, 0.04]])

    x = MinL1RiskModel(historical_returns, 0.07, 0.0, 1.0)
    x.addTurnoverLimit([0.33, 0.33, 0.33], 0.2)
    x.write()
    a = x.solve()
    print(a)
