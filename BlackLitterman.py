from cvxopt import matrix, solvers
from cvxopt.lapack import getrf, getri, getrs
from Markowitz import MeanVarOpt

def BlackLittermanEstimate(returns, covariance, views_portfolios, views_outcomes,
                           views_variance, return_confidence, risk_free_rate, market_weights, lagrange_multiplier):
    nsec = len(returns)

    #compute the inverse of the outcomes covariance
    nviews = views_portfolios.size[0]
    views_cov_inv = matrix(0.0, (nviews, nviews))
    for i in range(nviews):
        views_cov_inv[i,i] = 1.0 / views_variance[i]

    bl_returns = risk_free_rate * matrix(1.0, (nsec, 1)) + 2 * lagrange_multiplier * covariance * market_weights

    # compute the inverse
    cov_inv = covariance
    cov_perm = matrix(1, (nsec, 1))
    getrf(cov_inv, cov_perm)
    getri(cov_inv, cov_perm)

    # u = M^1 * b
    tmpmat = views_portfolios.T * views_cov_inv * views_portfolios
    M = cov_inv / return_confidence + tmpmat
    b = cov_inv * bl_returns / return_confidence + views_portfolios.T * views_cov_inv * views_outcomes

    m_perm = matrix(0, (nsec, nsec))
    getrf(M, m_perm)
    getrs(M, m_perm, b)

    # the covariance matrix
    bl_cov = covariance + cov_inv / return_confidence + views_portfolios.T * views_cov_inv * views_portfolios

    return (b, bl_cov)

if __name__ == "__main__":
    returns  = matrix([0.1073, 0.0737, 0.0627])
    covariance = matrix([[2.78e-02,  3.87e-03,  2.07e-04], [3.87e-03,  1.11e-02, -1.95e-04], [2.07e-04, -1.95e-04,  1.16e-03]])

    views_portfolios = matrix([1.0, -1.0, 0.0]).T
    views_outcomes = matrix([0.5])
    views_variance = matrix([0.15])
    return_confidence = 1.0 / 24.0
    risk_free_rate = 0.01
    market_weights = matrix([0.5, 0.4, 0.1])
    lagrange_multiplier = 1.25
    bl = BlackLittermanEstimate(returns, covariance,  views_portfolios, views_outcomes, views_variance, return_confidence, risk_free_rate, market_weights, lagrange_multiplier)
    #print(bl)

    hist_model = MeanVarOpt(returns, covariance, 0.07)
    print(hist_model.solve())

    bl_model = MeanVarOpt(bl[0], bl[1], 0.07)
    print(bl_model.solve())
