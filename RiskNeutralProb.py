from cvxopt import matrix, solvers
from math import pow

# TODO int p = 1

class RiskNeutralProbSpline:
    """Recover the risk neutral probabilities from call prices as cubic splines"""

    def __init__(self, call_strikes, call_prices, price_range, risk_free_rate):
        assert len(call_prices) == len(call_strikes)

        min_underlying_price = price_range[0]
        max_underlying_price = price_range[1]

        ninterv = len(call_prices) + 1

        # sort by ascending order
        pairs = []
        for i in range(len(call_prices)):
            pairs.append((call_strikes[i], call_prices[i]))
        takeFirst = lambda x : x[0]
        pairs.sort(key=takeFirst)

        # the objective coefficients for (alpha_k, beta_k, gamma_k, delta_k) 
        # corresoending to the strike k is given by the symertic matrix:
        # coef(k, 5)^2   coef(k, 5) * coef(k, 4)  coef(k, 5) * coef(k, 3) coef(k, 5) * coef(k,2)
        #     ---               coef(k,4)^2      coef(k, 4) * coef(k, 3) coef(k, 4) * coef(k,2)    
        #     ---                  ---                coef(k,3)^2        coef(k, 3) * coef(k,2)    
        #     ---                  ---                    ---                coef(k,2)^2
        #coef = lambda strike, i : ( (max_underlying_price ** i - strike ** i) / i - strike * (max_underlying_price ** (i-1) - strike ** (i-1)) / (i - 1) ) / (1 + risk_free_rate)
        def coef(strike, i):
            c1 = (max_underlying_price ** i - strike ** i) / i 
            c2 = (max_underlying_price ** (i-1) - strike ** (i-1)) / (i - 1)

            return (c1 - strike * c2) / (1 + risk_free_rate)

        # objective
        self.P = matrix(0.0, (4 * ninterv, 4 * ninterv))
        self.q = matrix(0.0, (4 * ninterv, 1))

        # equality constraints
        # for every knot
        # spline is continuous
        # split has a first derivative
        # split has a second derivative
        # the last 3 constraints
        # natural spline on min_price
        # natural spline on max_price
        # integral of prob = 1
        self.A = matrix(0.0, (3 * ninterv, 4 * ninterv))
        self.b = matrix(0.0, (3 * ninterv, 1))


        # TODO add non neg constraint on max price
        # inequality constraints
        # non-negativity on the nkots
        self.G = matrix(0.0, (ninterv + 2, 4 * ninterv))
        self.h = matrix(0.0, (ninterv + 2, 1))

        for i in range(ninterv-1):
            strike = pairs[i][0]
            price = pairs[i][1]

            coef_alpha = coef(strike, 5)
            coef_beta = coef(strike, 4)
            coef_gamma = coef(strike, 3)
            coef_delta = coef(strike, 2)

            # fill tself.he diagonal of tself.he oself.bjective
            # x - - -
            # - x - -
            # - - x -
            # - - - x
            self.P[4*i,4*i] += coef_alpha ** 2
            self.P[4*i+1,4*i+1] += coef_beta ** 2
            self.P[4*i+2,4*i+2] += coef_gamma ** 2
            self.P[4*i+3,4*i+3] += coef_delta ** 2
 
            # - x x x
            # x - - -
            # x - - -
            # x - - -
            self.P[4*i, 4*i+1] = coef_alpha * coef_beta
            self.P[4*i+1, 4*i] = self.P[4*i, 4*i+1]
            self.P[4*i, 4*i+2] = coef_alpha * coef_gamma
            self.P[4*i+2, 4*i] = self.P[4*i, 4*i+2]
            self.P[4*i, 4*i+3] = coef_alpha * coef_delta
            self.P[4*i+3, 4*i] = self.P[4*i, 4*i+3]
 
            # - - - -
            # - - x x
            # - x - -
            # - x - -
            self.P[4*i+1, 4*i+2] = coef_beta * coef_gamma
            self.P[4*i+2, 4*i+1] = self.P[4*i+1, 4*i+2]
            self.P[4*i+1, 4*i+3] = coef_beta * coef_delta
            self.P[4*i+3, 4*i+1] = self.P[4*i+1, 4*i+3]
 
            # - - - -
            # - - - -
            # - - - x
            # - - x -
            self.P[4*i+2, 4*i+3] = coef_gamma * coef_delta
            self.P[4*i+3, 4*i+2] = self.P[4*i+2, 4*i+3]

            # linear oself.bjective
            self.q[4*i] = -2 * price * coef_alpha
            self.q[4*i+1] = -2 * price * coef_beta
            self.q[4*i+2] = -2 * price * coef_gamma
            self.q[4*i+3] = -2 * price * coef_delta

            # spline(knot_(i+1)) = spline(knot_i) 
            row = 3*i
            self.A[row, 4*i] = strike ** 3
            self.A[row, 4*i+1] = strike ** 2
            self.A[row, 4*i+2] = strike
            self.A[row, 4*i+3] = 1.0
            self.A[row, 4*(i+1)] = -strike ** 3
            self.A[row, 4*(i+1)+1] = -strike ** 2
            self.A[row, 4*(i+1)+2] = -strike
            self.A[row, 4*(i+1)+3] = -1.0

            # d[spline(knot_(i+1))]/d[strike] = d[spline(knot_i)]/d[strike]
            row += 1
            self.A[row, 4*i] = 3 * strike ** 2
            self.A[row, 4*i+1] = 2 * strike
            self.A[row, 4*i+2] = 1.0
            self.A[row, 4*(i+1)] = -3 * strike ** 2
            self.A[row, 4*(i+1)+1] = -2 * strike
            self.A[row, 4*(i+1)+2] = -1.0

            # d2[spline(knot_(i+1))]/d[strike]2 = d2[spline(knot_i)]/d[strike]2
            row += 1
            self.A[row, 4*i] = 6 * strike
            self.A[row, 4*i+1] = 2.0
            self.A[row, 4*(i+1)] = -6 * strike
            self.A[row, 4*(i+1)+1] = -2.0

        for i in range(ninterv):
            # nonnegativity contraint on knots
            self.G[i, 4*i] = -strike ** 3
            self.G[i, 4*i+1] = -strike ** 2
            self.G[i, 4*i+2] = -strike
            self.G[i, 4*i+3] = -1.0

        # unit integral constraint
        unitintrow = 3*ninterv - 1
        self.b[unitintrow] = 1.0
        for i in range(ninterv):
            rightpt = -1.0
            leftpt = -1.0

            if i == ninterv-1:
                rightpt = max_underlying_price
            else:
                rightpt = pairs[i][0]

            if i == 0:
                leftpt = min_underlying_price
            else:
                leftpt = pairs[i-1][0]

            assert rightpt > 0.0
            assert leftpt > 0.0

            self.A[unitintrow, 4*(i-1)] = (rightpt**4 - leftpt**4)/4
            self.A[unitintrow, 4*(i-1)+1] = (rightpt**3 - leftpt**3)/3
            self.A[unitintrow, 4*(i-1)+2] =  (rightpt**2 - leftpt**2)/2
            self.A[unitintrow, 4*(i-1)+3] =  rightpt - leftpt

        # natural spline in min price
        beforelastrowid = 3 * ninterv - 2
        self.A[beforelastrowid, 0] = 3 * min_underlying_price ** 2
        self.A[beforelastrowid, 1] = 2 * min_underlying_price
        self.A[beforelastrowid, 2] = 1.0

        # natural spline on max price
        lastrowid = 3 * ninterv - 3
        self.A[lastrowid, 4*ninterv-4] = 3 * max_underlying_price ** 2
        self.A[lastrowid, 4*ninterv-3] = 2 * max_underlying_price
        self.A[lastrowid, 4*ninterv-2] = 1.0

        # nongeative spline on min price
        beforelastnngcons = ninterv
        self.G[beforelastnngcons, 0] = -min_underlying_price ** 3
        self.G[beforelastnngcons, 1] = -min_underlying_price ** 2
        self.G[beforelastnngcons, 2] = -min_underlying_price
        self.G[beforelastnngcons, 3] = -1.0

        # nongeative spline on max price
        lastnngcons = ninterv + 1
        self.G[lastnngcons, 4*(ninterv-1)] = -max_underlying_price ** 3
        self.G[lastnngcons, 4*(ninterv-1)+1] = -max_underlying_price ** 2
        self.G[lastnngcons, 4*(ninterv-1)+2] = -max_underlying_price
        self.G[lastnngcons, 4*(ninterv-1)+3] = -1.0
 
    def write(self):
        print("inequality constraints Gx <= h:")
        for i in range(self.G.size[0]):
            cons = "[{}]: ".format(i+1)
            for j in range(self.G.size[1]):
                if self.G[i,j] > 0.0:
                        cons += " + {}<{}>".format(self.G[i,j], j+1)
                elif self.G[i,j] < 0.0:
                        cons += " - {}<{}>".format(-self.G[i,j], j+1)
            cons += " <= {}".format(self.h[i])
            print(cons)
    
        print("equality constraints Ax = b")
        for i in range(self.A.size[0]):
            cons = "[{}]: ".format(i+1)
            for j in range(self.A.size[1]):
                if self.A[i,j] > 0.0:
                        cons += " + {}<{}>".format(self.A[i,j], j+1)
                elif self.A[i,j] < 0.0:
                        cons += " - {}<{}>".format(-self.A[i,j], j+1)
            cons += " = {}".format(self.b[i])
            print(cons)
    
        for i in range(self.P.size[0]):
            str = ""
            for j in range(self.P.size[1]):
                if self.P[i,j] != 0.0:
                    str += "x "
                else:
                    str += "- "
        print(str)

    def solve(self):
        sol = solvers.qp(self.P, self.q, self.G, self.h, self.A, self.b)
        return sol['x']

call_strikes = [2.0, 15.0, 23.0, 50.0]
call_prices = [1.0, 9.0, 10.0, 40.0]
x = RiskNeutralProbSpline(call_strikes, call_prices, [0.5, 30], 0.01)
x.write()
sol = x.solve()
print(sol)

pol = lambda a, b, c, d, x : a*x**3 + b*x**2 + c*x + d
strike = call_strikes[1]
print("strike {} prob {}".format(strike, pol(sol[0], sol[1], sol[2], sol[3], strike)))
strike = call_strikes[2]
print("strike {} prob {}".format(strike, pol(sol[4], sol[5], sol[6], sol[7], strike)))
