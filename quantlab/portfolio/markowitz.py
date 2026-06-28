"""
Markowitz Mean-Variance Portfolio Optimization.

Implements the classic Markowitz framework with closed-form solutions
using KKT (Karush-Kuhn-Tucker) conditions. No external optimization
library (cvxopt, scipy) is required.

All solutions are analytically derived for the unconstrained and
constrained optimization problems.

References
----------
Markowitz, H. (1952). "Portfolio Selection." Journal of Finance, 7(1), 77-91.
Merton, R.C. (1972). "An Analytic Derivation of the Efficient Portfolio Frontier."
Journal of Financial and Quantitative Analysis, 7(4), 1851-1872.
"""

import numpy as np


def efficient_frontier(expected_returns, cov_matrix, n_points=100, rf=0.0,
                        allow_short=True):
    r"""
    Compute the Markowitz efficient frontier.

    Parameters
    ----------
    expected_returns : ndarray of shape (n_assets,)
        Vector of expected returns for each asset.
    cov_matrix : ndarray of shape (n_assets, n_assets)
        Covariance matrix of asset returns.
    n_points : int, optional
        Number of points on the frontier (default 100).
    rf : float, optional
        Risk-free rate (default 0).
    allow_short : bool, optional
        If True, allow short selling (default True). If False, imposes
        :math:`w_i \ge 0`.

    Returns
    -------
    returns : ndarray
        Expected returns of frontier portfolios.
    volatilities : ndarray
        Standard deviations of frontier portfolios.
    weights : ndarray of shape (n_points, n_assets)
        Portfolio weights for each point on the frontier.

    Notes
    -----
    For the unconstrained case, the efficient frontier is derived from the
    two-fund separation theorem. Any efficient portfolio is a combination of:

    1. The global minimum variance portfolio (GMVP)
    2. A second efficient portfolio

    The KKT conditions for :math:`\min_w \frac{1}{2}w^T\Sigma w` subject to
    :math:`w^T\mathbf{1}=1` and :math:`w^T\mu = \mu_p` yield closed-form solutions.

    For the long-only case, we sample along fixed grid points using the
    Lagrangian with inequality constraints.
    """
    n = len(expected_returns)
    expected_returns = np.asarray(expected_returns, dtype=float)
    cov_matrix = np.asarray(cov_matrix, dtype=float)

    ones = np.ones(n)
    inv_cov = np.linalg.inv(cov_matrix)

    if allow_short:
        # Unconstrained: two-fund separation

        # GMVP
        A = ones @ inv_cov @ ones
        w_gmv = inv_cov @ ones / A
        ret_gmv = expected_returns @ w_gmv
        var_gmv = w_gmv @ cov_matrix @ w_gmv

        # Second efficient portfolio
        B = expected_returns @ inv_cov @ ones
        C = expected_returns @ inv_cov @ expected_returns

        # GMVP and excess-return portfolio
        z = inv_cov @ (expected_returns - rf * ones)
        w_tan = z / (ones @ z)
        ret_tan = expected_returns @ w_tan
        var_tan = w_tan @ cov_matrix @ w_tan

        # Parameterize the frontier
        target_returns = np.linspace(ret_gmv, max(ret_tan, ret_gmv) * 1.5, n_points)
        weights = np.zeros((n_points, n))
        rets = np.zeros(n_points)
        vols = np.zeros(n_points)

        for i, target in enumerate(target_returns):
            w, ret, vol = _solve_mv_target(expected_returns, cov_matrix, inv_cov,
                                            ones, target, A, B, C)
            weights[i] = w
            rets[i] = ret
            vols[i] = vol

        return rets, vols, weights
    else:
        # Long-only: sample GMV to max return along constrained frontier
        # Use a grid sampling approach with iterative refinement
        w_gmv_l = _long_only_gmv(cov_matrix)
        ret_gmv = expected_returns @ w_gmv_l
        ret_max = np.max(expected_returns)

        target_returns = np.linspace(ret_gmv, ret_max * 0.95, n_points)
        weights = np.zeros((n_points, n))
        rets = np.zeros(n_points)
        vols = np.zeros(n_points)

        # For long-only, we use a simple coordinate-wise optimization per target
        for i, target in enumerate(target_returns):
            w = _long_only_target(expected_returns, cov_matrix, target)
            weights[i] = w
            rets[i] = expected_returns @ w
            vols[i] = np.sqrt(w @ cov_matrix @ w)

        return rets, vols, weights


def _solve_mv_target(expected_returns, cov, inv_cov, ones, target_return, A, B, C):
    """Solve mean-variance problem for a target return (unconstrained, closed-form)."""
    # Lagrangian multipliers
    lambda1 = (C - target_return * B) / (A * C - B ** 2)
    lambda2 = (target_return * A - B) / (A * C - B ** 2)

    w = lambda1 * (inv_cov @ ones) + lambda2 * (inv_cov @ expected_returns)
    ret_val = w @ expected_returns
    vol_val = np.sqrt(w @ cov @ w)

    return w, ret_val, vol_val


def _long_only_gmv(cov):
    """Compute long-only global minimum variance portfolio via gradient projection."""
    n = len(cov)
    w = np.ones(n) / n
    lr = 0.01

    for _ in range(500):
        grad = cov @ w
        w_new = w - lr * grad
        # Project to simplex (sum=1, w>=0)
        w_new = np.maximum(w_new, 0)
        w_new = w_new / w_new.sum()
        w = w_new

    return w


def _long_only_target(expected_returns, cov, target_return):
    """Compute long-only portfolio for target return."""
    n = len(expected_returns)
    w = np.ones(n) / n

    for _ in range(1000):
        # Gradient of variance + penalty for return deviation
        grad_var = cov @ w
        current_ret = w @ expected_returns
        penalty = 2.0 * max(0, target_return - current_ret)
        grad_penalty = -penalty * expected_returns

        w_new = w - 0.001 * (grad_var + grad_penalty)
        w_new = np.maximum(w_new, 0)
        if w_new.sum() > 0:
            w_new = w_new / w_new.sum()
        else:
            w_new = np.ones(n) / n

        if np.max(np.abs(w_new - w)) < 1e-8:
            break
        w = w_new

    return w


def min_variance_portfolio(cov_matrix):
    r"""
    Compute the global minimum variance portfolio (GMVP).

    Parameters
    ----------
    cov_matrix : ndarray of shape (n_assets, n_assets)
        Covariance matrix of asset returns.

    Returns
    -------
    ndarray
        GMVP weights.

    Notes
    -----
    The GMVP solves:

    .. math::
        \min_w \frac{1}{2} w^T \Sigma w \quad \text{s.t.} \quad w^T \mathbf{1} = 1

    The closed-form solution is:

    .. math::
        w_{GMVP} = \frac{\Sigma^{-1}\mathbf{1}}{\mathbf{1}^T\Sigma^{-1}\mathbf{1}}
    """
    cov = np.asarray(cov_matrix, dtype=float)
    n = len(cov)
    ones = np.ones(n)

    try:
        inv_cov = np.linalg.inv(cov)
        w = inv_cov @ ones / (ones @ inv_cov @ ones)
    except np.linalg.LinAlgError:
        # Fallback: pseudoinverse
        inv_cov = np.linalg.pinv(cov)
        w = inv_cov @ ones / (ones @ inv_cov @ ones)

    return w


def tangency_portfolio(expected_returns, cov_matrix, rf=0.0):
    r"""
    Compute the tangency (maximum Sharpe ratio) portfolio.

    Parameters
    ----------
    expected_returns : ndarray
        Expected returns.
    cov_matrix : ndarray
        Covariance matrix.
    rf : float, optional
        Risk-free rate (default 0).

    Returns
    -------
    ndarray
        Tangency portfolio weights.

    Notes
    -----
    The tangency portfolio solves:

    .. math::
        \max_w \frac{w^T\mu - r_f}{\sqrt{w^T\Sigma w}}
        \quad \text{s.t.} \quad w^T\mathbf{1} = 1

    The unconstrained solution is:

    .. math::
        w_{\text{tan}}^* = \frac{\Sigma^{-1}(\mu - r_f\mathbf{1})}
        {\mathbf{1}^T\Sigma^{-1}(\mu - r_f\mathbf{1})}
    """
    expected_returns = np.asarray(expected_returns, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)
    n = len(expected_returns)
    ones = np.ones(n)
    excess = expected_returns - rf * ones

    try:
        inv_cov = np.linalg.inv(cov)
        w_raw = inv_cov @ excess
        denom = ones @ w_raw
        w = w_raw / denom if abs(float(denom)) > 1e-15 else ones / n
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)
        w_raw = inv_cov @ excess
        denom = ones @ w_raw
        w = w_raw / denom if denom != 0 else ones / n

    return w


def max_sharpe_portfolio(expected_returns, cov_matrix, rf=0.0):
    r"""
    Compute the maximum Sharpe ratio portfolio (same as tangency portfolio).

    Parameters
    ----------
    expected_returns : ndarray
        Expected returns.
    cov_matrix : ndarray
        Covariance matrix.
    rf : float, optional
        Risk-free rate (default 0).

    Returns
    -------
    ndarray
        Max Sharpe portfolio weights.
    """
    return tangency_portfolio(expected_returns, cov_matrix, rf)


def portfolio_stats(weights, expected_returns, cov_matrix, rf=0.0):
    r"""
    Compute portfolio return, volatility, and Sharpe ratio.

    Parameters
    ----------
    weights : ndarray
        Portfolio weights.
    expected_returns : ndarray
        Expected asset returns.
    cov_matrix : ndarray
        Covariance matrix.
    rf : float, optional
        Risk-free rate (default 0).

    Returns
    -------
    dict
        Dictionary with 'return', 'volatility', 'sharpe_ratio'.
    """
    weights = np.asarray(weights, dtype=float)
    expected_returns = np.asarray(expected_returns, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)

    port_return = weights @ expected_returns
    port_vol = np.sqrt(weights @ cov @ weights)
    sharpe = (port_return - rf) / port_vol if port_vol > 0 else 0.0

    return {
        'return': port_return,
        'volatility': port_vol,
        'sharpe_ratio': sharpe,
    }
