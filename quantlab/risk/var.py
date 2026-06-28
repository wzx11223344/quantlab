"""
Value at Risk (VaR) and Conditional VaR (CVaR).

Implements three VaR methodologies (Historical, Parametric, Monte Carlo),
Conditional VaR (Expected Shortfall), and VaR backtesting with the Kupiec
POF test and Christoffersen independence test.

References
----------
Jorion, P. (2007). "Value at Risk: The New Benchmark for Managing Financial Risk."
Kupiec, P.H. (1995). "Techniques for Verifying the Accuracy of Risk Measurement Models."
Christoffersen, P.F. (1998). "Evaluating Interval Forecasts."
"""

import numpy as np
from ..utils import norm_ppf, norm_cdf


def historical_var(returns, alpha=0.05):
    r"""
    Compute Historical Value at Risk (VaR) using empirical quantiles.

    Parameters
    ----------
    returns : array_like
        Array of historical returns.
    alpha : float, optional
        Significance level (default 0.05 for 95% VaR).

    Returns
    -------
    float
        The historical VaR at the given confidence level.

    Notes
    -----
    :math:`\text{VaR}_\alpha = -Q_\alpha(\text{returns})`

    where :math:`Q_\alpha` is the :math:`\alpha`-quantile of the return
    distribution. The negative sign ensures VaR is reported as a positive
    loss amount.

    Examples
    --------
    >>> returns = np.random.normal(0.001, 0.02, 1000)
    >>> historical_var(returns, alpha=0.05)  # 95% VaR
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 2:
        return 0.0
    return -np.quantile(returns, alpha)


def parametric_var(returns, alpha=0.05):
    r"""
    Compute Parametric (Variance-Covariance) VaR assuming normality.

    Parameters
    ----------
    returns : array_like
        Array of historical returns.
    alpha : float, optional
        Significance level (default 0.05 for 95% VaR).

    Returns
    -------
    float
        Parametric VaR.

    Notes
    -----
    Under the normality assumption:

    .. math::
        \text{VaR}_\alpha = -(\mu + z_\alpha \sigma)

    where :math:`\mu` is the mean return, :math:`\sigma` is the standard
    deviation, and :math:`z_\alpha = \Phi^{-1}(\alpha)` is the quantile
    of the standard normal distribution.
    """
    returns = np.asarray(returns, dtype=float)
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    z_alpha = norm_ppf(alpha)
    return -(mu + z_alpha * sigma)


def monte_carlo_var(initial_value, mu, sigma, n_sims=10000, alpha=0.05, horizon=1, seed=None):
    r"""
    Compute Monte Carlo VaR by simulating future returns.

    Parameters
    ----------
    initial_value : float
        Initial portfolio value.
    mu : float
        Expected return (annualized).
    sigma : float
        Volatility (annualized).
    n_sims : int, optional
        Number of simulation paths (default 10000).
    alpha : float, optional
        Significance level (default 0.05).
    horizon : float, optional
        Time horizon in years (default 1).
    seed : int, optional
        Random seed.

    Returns
    -------
    float
        Monte Carlo VaR (positive loss amount).

    Notes
    -----
    Simulates :math:`n` paths of the portfolio value:

    .. math::
        V_T = V_0 \exp\left((\mu - \tfrac{1}{2}\sigma^2)T + \sigma\sqrt{T}Z\right)

    VaR is the :math:`\alpha`-quantile of the loss distribution:

    .. math::
        \text{VaR}_\alpha = V_0 - Q_\alpha(V_T)
    """
    if seed is not None:
        np.random.seed(seed)

    Z = np.random.standard_normal(n_sims)
    drift = (mu - 0.5 * sigma ** 2) * horizon
    vol_sqrt = sigma * np.sqrt(horizon)

    final_values = initial_value * np.exp(drift + vol_sqrt * Z)
    losses = initial_value - final_values

    return np.quantile(losses, 1.0 - alpha)


def cvar(returns, alpha=0.05):
    r"""
    Compute Conditional Value at Risk (CVaR / Expected Shortfall).

    CVaR is the expected loss given that the loss exceeds VaR.

    Parameters
    ----------
    returns : array_like
        Array of returns.
    alpha : float, optional
        Significance level (default 0.05 for 95% CVaR).

    Returns
    -------
    float
        Conditional VaR (positive loss amount).

    Notes
    -----
    .. math::
        \text{CVaR}_\alpha = -\mathbb{E}[R \mid R \leq Q_\alpha(R)]

    where :math:`Q_\alpha` is the :math:`\alpha`-quantile of returns.
    This is the expected shortfall: the average of returns worse than VaR.

    For the normal distribution, the analytical CVaR is:

    .. math::
        \text{CVaR}_\alpha = - \mu + \sigma \cdot
        \frac{\phi(z_\alpha)}{\alpha}
    """
    returns = np.asarray(returns, dtype=float)
    var_threshold = -historical_var(returns, alpha)
    tail = returns[returns <= var_threshold]
    if len(tail) == 0:
        return 0.0
    return -np.mean(tail)


def backtest_var(returns, var_forecasts, alpha=0.05):
    r"""
    Backtest VaR forecasts using Kupiec POF and Christoffersen tests.

    Parameters
    ----------
    returns : array_like
        Realized returns (length n).
    var_forecasts : array_like
        VaR forecasts (length n, as positive loss amounts).
        Each element is VaR_{t} forecast for period t+1.
    alpha : float, optional
        Significance level used for VaR (default 0.05).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'n_exceedances': number of VaR violations
        - 'expected_exceedances': expected violations under correct model
        - 'violation_rate': actual violation rate
        - 'kupiec_stat': Kupiec POF test statistic
        - 'kupiec_pvalue': p-value for Kupiec test
        - 'christoffersen_stat': Christoffersen test statistic
        - 'christoffersen_pvalue': p-value

    Notes
    -----
    **Kupiec POF (Proportion of Failures) Test:**

    Tests if the observed violation rate equals the expected rate :math:`\alpha`.

    .. math::
        LR_{POF} = -2\log\left[\frac{(1-\alpha)^{n-x}\alpha^x}
        {(1-\hat{\pi})^{n-x}\hat{\pi}^x}\right] \sim \chi^2_1

    where :math:`\hat{\pi} = x/n` is the observed violation rate.

    **Christoffersen Independence Test:**

    Tests if violations are clustered (violation today affects probability
    of violation tomorrow).

    .. math::
        LR_{CC} = LR_{POF} + LR_{ind}
    """
    returns = np.asarray(returns, dtype=float)
    var_forecasts = np.asarray(var_forecasts, dtype=float)

    if len(returns) != len(var_forecasts):
        raise ValueError("returns and var_forecasts must have the same length")

    n = len(returns)
    var_forecasts = np.abs(var_forecasts)

    # Violations: return < -VaR (loss exceeds VaR)
    violations = (returns < -var_forecasts).astype(int)
    x = int(violations.sum())
    expected_x = alpha * n
    violation_rate = x / n if n > 0 else 0.0

    # Kupiec POF test
    if x == 0 or x == n:
        # Edge case: use approximation
        kupiec_stat = np.nan
        kupiec_pvalue = np.nan
    else:
        pi_hat = x / n
        # LR statistic
        ll_unrestricted = x * np.log(pi_hat) + (n - x) * np.log(1 - pi_hat)
        ll_restricted = x * np.log(alpha) + (n - x) * np.log(1 - alpha)
        kupiec_stat = -2.0 * (ll_restricted - ll_unrestricted)
        # Chi-squared(1) p-value
        kupiec_pvalue = 1.0 - _chi2_cdf(kupiec_stat, 1)

    # Christoffersen independence test
    # Transition counts
    n00 = n01 = n10 = n11 = 0
    for t in range(n - 1):
        if violations[t] == 0 and violations[t + 1] == 0:
            n00 += 1
        elif violations[t] == 0 and violations[t + 1] == 1:
            n01 += 1
        elif violations[t] == 1 and violations[t + 1] == 0:
            n10 += 1
        elif violations[t] == 1 and violations[t + 1] == 1:
            n11 += 1

    if n00 + n01 == 0 or n10 + n11 == 0:
        # Not enough transitions
        christoffersen_stat = np.nan
        christoffersen_pvalue = np.nan
    else:
        pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
        pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
        pi = (n01 + n11) / n if n > 0 else 0.0

        # Independence LR
        if pi0 > 0 and pi0 < 1 and pi1 > 0 and pi1 < 1 and pi > 0 and pi < 1:
            ll_indep = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
            ll_dep = (
                n00 * np.log(1 - pi0) + n01 * np.log(pi0)
                + n10 * np.log(1 - pi1) + n11 * np.log(pi1)
            )
            lr_ind = -2.0 * (ll_indep - ll_dep)
        else:
            lr_ind = 0.0

        # Conditional coverage: POF + independence
        if not np.isnan(kupiec_stat):
            christoffersen_stat = kupiec_stat + lr_ind
            christoffersen_pvalue = 1.0 - _chi2_cdf(christoffersen_stat, 2)
        else:
            christoffersen_stat = lr_ind
            christoffersen_pvalue = 1.0 - _chi2_cdf(christoffersen_stat, 1)

    return {
        'n_exceedances': x,
        'expected_exceedances': expected_x,
        'violation_rate': violation_rate,
        'kupiec_stat': kupiec_stat,
        'kupiec_pvalue': kupiec_pvalue,
        'christoffersen_stat': christoffersen_stat,
        'christoffersen_pvalue': christoffersen_pvalue,
    }


def _chi2_cdf(x, df):
    """
    Compute the CDF of the chi-squared distribution with df degrees of freedom.

    Uses the regularized lower incomplete gamma function.
    """
    if x <= 0:
        return 0.0
    return _gammainc(df / 2.0, x / 2.0)


def _gammainc(a, x):
    """
    Regularized lower incomplete gamma function P(a, x).

    Uses series expansion for small x, continued fraction for large x.
    """
    if x < 0:
        return 0.0
    if x == 0:
        return 0.0
    if a <= 0:
        return 1.0

    # Series expansion
    from math import exp, log, gamma, lgamma

    if x < a + 1:
        # Series
        log_gamma_a = lgamma(a)
        sum_val = 1.0 / a
        term = 1.0 / a
        for n in range(1, 200):
            term *= x / (a + n)
            old_sum = sum_val
            sum_val += term
            if abs(sum_val - old_sum) < 1e-15:
                break
        result = sum_val * exp(-x + a * log(x) - log_gamma_a)
    else:
        # Continued fraction (Lentz's method)
        f = 1e-30
        c_val = 1e-30
        d = 0.0
        for n in range(1, 200):
            if n % 2 == 1:
                d = x + n - a + (n * d) if d != 0 else x + n - a + n * 1e-30
            else:
                d = 1.0 + n * (a - n) / d if d != 0 else 1e-30
            c_val = x + n - a + n / c_val if c_val != 0 else x + n - a + n * 1e-30
            if d != 0:
                d = 1.0 / d
            delta = c_val * d
            f *= delta
            if abs(delta - 1.0) < 1e-15:
                break

        log_gamma_a = lgamma(a)
        result = 1.0 - exp(-x + a * log(x) - log_gamma_a) / f

    return max(0.0, min(1.0, result))
