"""
Utility functions for QuantLab.

Provides Brownian motion simulation, Heston path simulation, log returns,
and normality tests. All implementations are pure NumPy.
"""

import numpy as np


def simulate_gbm(S0, mu, sigma, T, N, n_paths=1, seed=None):
    """
    Simulate Geometric Brownian Motion paths.

    Parameters
    ----------
    S0 : float
        Initial asset price.
    mu : float
        Drift (annualized).
    sigma : float
        Volatility (annualized).
    T : float
        Time horizon in years.
    N : int
        Number of time steps.
    n_paths : int, optional
        Number of simulated paths (default 1).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ndarray of shape (N + 1, n_paths)
        Simulated price paths. First row is S0.

    Notes
    -----
    Under the risk-neutral measure, set mu = r (risk-free rate).
    The GBM SDE is:

    .. math::
        dS_t = \\mu S_t dt + \\sigma S_t dW_t

    Discretized via Euler-Maruyama:

    .. math::
        S_{t+\\Delta t} = S_t \\exp\\left((\\mu - \\tfrac{1}{2}\\sigma^2)\\Delta t
        + \\sigma \\sqrt{\\Delta t} Z\\right), \\quad Z \\sim \\mathcal{N}(0,1)
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / N
    drift = (mu - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    paths = np.zeros((N + 1, n_paths))
    paths[0, :] = S0

    for t in range(1, N + 1):
        Z = np.random.standard_normal(n_paths)
        paths[t, :] = paths[t - 1, :] * np.exp(drift + diffusion * Z)

    return paths


def simulate_heston(S0, v0, kappa, theta, xi, rho, T, N, n_paths=1, seed=None):
    """
    Simulate asset paths under the Heston stochastic volatility model.

    Parameters
    ----------
    S0 : float
        Initial asset price.
    v0 : float
        Initial variance.
    kappa : float
        Mean reversion speed of variance.
    theta : float
        Long-run mean variance.
    xi : float
        Volatility of variance (vol-of-vol).
    rho : float
        Correlation between asset and variance Brownian motions.
    T : float
        Time horizon in years.
    N : int
        Number of time steps.
    n_paths : int, optional
        Number of simulated paths (default 1).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    S : ndarray of shape (N + 1, n_paths)
        Simulated price paths.
    v : ndarray of shape (N + 1, n_paths)
        Simulated variance paths.

    Notes
    -----
    The Heston model SDEs are:

    .. math::
        dS_t &= \\mu S_t dt + \\sqrt{v_t} S_t dW_t^{(1)} \\\\
        dv_t &= \\kappa(\\theta - v_t) dt + \\xi \\sqrt{v_t} dW_t^{(2)} \\\\
        \\text{Corr}(dW_t^{(1)}, dW_t^{(2)}) &= \\rho

    Uses full truncation scheme to handle Feller condition violations:
    :math:`v_t = \\max(v_t, 0)`.
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / N

    S = np.zeros((N + 1, n_paths))
    v = np.zeros((N + 1, n_paths))
    S[0, :] = S0
    v[0, :] = v0

    for t in range(1, N + 1):
        # Generate correlated Brownian increments
        Z1 = np.random.standard_normal(n_paths)
        Z2 = rho * Z1 + np.sqrt(1 - rho ** 2) * np.random.standard_normal(n_paths)

        # Full truncation: ensure variance is non-negative
        v_prev = np.maximum(v[t - 1, :], 0.0)
        sqrt_v = np.sqrt(v_prev)

        # Variance process
        v[t, :] = v_prev + kappa * (theta - v_prev) * dt + xi * sqrt_v * np.sqrt(dt) * Z2

        # Asset price process
        S[t, :] = S[t - 1, :] * np.exp(
            -0.5 * v_prev * dt + sqrt_v * np.sqrt(dt) * Z1
        )

    return S, v


def log_returns(prices):
    """
    Compute logarithmic returns from a price series.

    Parameters
    ----------
    prices : array_like
        Price series.

    Returns
    -------
    ndarray
        Log returns: :math:`r_t = \\log(P_t / P_{t-1})`.

    Notes
    -----
    The output has length ``len(prices) - 1``.
    """
    prices = np.asarray(prices, dtype=float)
    return np.log(prices[1:] / prices[:-1])


def simple_returns(prices):
    """
    Compute simple (arithmetic) returns from a price series.

    Parameters
    ----------
    prices : array_like
        Price series.

    Returns
    -------
    ndarray
        Simple returns: :math:`R_t = (P_t - P_{t-1}) / P_{t-1}`.
    """
    prices = np.asarray(prices, dtype=float)
    return (prices[1:] - prices[:-1]) / prices[:-1]


def norm_pdf(x):
    """
    Standard normal probability density function.

    Parameters
    ----------
    x : float or array_like
        Input value(s).

    Returns
    -------
    float or ndarray
        PDF value(s): :math:`\\phi(x) = \\frac{1}{\\sqrt{2\\pi}} e^{-x^2/2}`.
    """
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x ** 2)


def norm_cdf(x):
    """
    Standard normal cumulative distribution function.

    Uses the error function approximation (no scipy dependency).

    Parameters
    ----------
    x : float or array_like
        Input value(s).

    Returns
    -------
    float or ndarray
        CDF value(s): :math:`\\Phi(x) = \\frac{1}{2}[1 + \\text{erf}(x/\\sqrt{2})]`.

    Notes
    -----
    The error function is approximated using the Abramowitz and Stegun
    formula 7.1.26 with maximum error 1.5e-7.
    """
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def erf(x):
    """
    Error function approximation (Abramowitz & Stegun 7.1.26).

    Maximum error: 1.5e-7.

    Parameters
    ----------
    x : float or array_like
        Input value(s).

    Returns
    -------
    float or ndarray
        erf(x) values.
    """
    x = np.asarray(x, dtype=float)
    sign = np.sign(x)
    x = np.abs(x)

    # Constants for the rational approximation
    p = 0.3275911
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)

    return sign * y


def erfinv(y):
    """
    Inverse error function approximation.

    Uses the method from Giles (2016) for high accuracy.

    Parameters
    ----------
    y : float or array_like
        Input value(s) in (-1, 1).

    Returns
    -------
    float or ndarray
        erf^{-1}(y) values.

    Raises
    ------
    ValueError
        If any |y| >= 1.
    """
    y = np.asarray(y, dtype=float)
    if np.any(np.abs(y) >= 1.0):
        raise ValueError("erfinv argument must be in (-1, 1)")

    # Constants for approximation
    a = 0.147
    ln_term = np.log(1.0 - y * y)
    term1 = 2.0 / (np.pi * a) + 0.5 * ln_term
    term2 = ln_term / a

    result = np.sign(y) * np.sqrt(np.sqrt(term1 * term1 - term2) - term1)

    # Newton refinement for accuracy
    for _ in range(2):
        err = erf(result) - y
        result -= err / (2.0 / np.sqrt(np.pi) * np.exp(-result * result))

    return result


def norm_ppf(q):
    """
    Standard normal percent point function (inverse CDF).

    Parameters
    ----------
    q : float or array_like
        Quantile(s) in (0, 1).

    Returns
    -------
    float or ndarray
        :math:`\\Phi^{-1}(q)` values.
    """
    return np.sqrt(2.0) * erfinv(2.0 * q - 1.0)


def jarque_bera(returns):
    """
    Jarque-Bera test for normality.

    Tests the null hypothesis that the data is normally distributed against
    the alternative that it is not.

    Parameters
    ----------
    returns : array_like
        Return series.

    Returns
    -------
    jb_stat : float
        Jarque-Bera test statistic.
    p_value : float
        p-value from chi-squared distribution with 2 degrees of freedom.

    Notes
    -----
    The test statistic is:

    .. math::
        JB = \\frac{n}{6}\\left(S^2 + \\frac{(K - 3)^2}{4}\\right)

    where *S* is sample skewness, *K* is sample kurtosis, and *n* is sample size.
    Under the null, :math:`JB \\sim \\chi^2_2` asymptotically.
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)

    if n < 4:
        return 0.0, 1.0

    # Compute centered moments
    mean = np.mean(returns)
    m2 = np.mean((returns - mean) ** 2)
    m3 = np.mean((returns - mean) ** 3)
    m4 = np.mean((returns - mean) ** 4)

    if m2 <= 0:
        return np.inf, 0.0

    skewness = m3 / (m2 ** 1.5)
    kurtosis = m4 / (m2 ** 2)

    jb_stat = n / 6.0 * (skewness ** 2 + (kurtosis - 3.0) ** 2 / 4.0)

    # Chi-squared(2) tail: P(X > x) for chi2(2) = exp(-x/2)
    p_value = np.exp(-jb_stat / 2.0) if jb_stat > 0 else 1.0

    return jb_stat, p_value


def box_muller(n=1, seed=None):
    """
    Generate standard normal random numbers using the Box-Muller transform.

    Parameters
    ----------
    n : int
        Number of random numbers to generate.
    seed : int, optional
        Random seed.

    Returns
    -------
    ndarray
        Array of n standard normal random numbers.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate pairs
    n_pairs = (n + 1) // 2
    U1 = np.random.uniform(0.0, 1.0, n_pairs)
    U2 = np.random.uniform(0.0, 1.0, n_pairs)

    # Avoid log(0)
    U1 = np.maximum(U1, 1e-15)

    R = np.sqrt(-2.0 * np.log(U1))
    theta = 2.0 * np.pi * U2

    Z1 = R * np.cos(theta)
    Z2 = R * np.sin(theta)

    Z = np.concatenate([Z1, Z2])
    return Z[:n]


def annualize_return(returns, periods_per_year=252):
    """
    Annualize returns.

    Parameters
    ----------
    returns : array_like
        Periodic returns.
    periods_per_year : int
        Number of periods per year (252 for daily, 12 for monthly).

    Returns
    -------
    float
        Annualized return.
    """
    returns = np.asarray(returns, dtype=float)
    total_return = np.prod(1.0 + returns)
    n_years = len(returns) / periods_per_year
    if n_years <= 0:
        return 0.0
    return total_return ** (1.0 / n_years) - 1.0


def annualize_volatility(returns, periods_per_year=252):
    """
    Annualize volatility from periodic returns.

    Parameters
    ----------
    returns : array_like
        Periodic returns.
    periods_per_year : int
        Number of periods per year.

    Returns
    -------
    float
        Annualized volatility.
    """
    returns = np.asarray(returns, dtype=float)
    return np.std(returns, ddof=1) * np.sqrt(periods_per_year)


def covariance_to_correlation(cov):
    """
    Convert a covariance matrix to a correlation matrix.

    Parameters
    ----------
    cov : ndarray
        Covariance matrix.

    Returns
    -------
    ndarray
        Correlation matrix.
    """
    cov = np.asarray(cov, dtype=float)
    std = np.sqrt(np.diag(cov))
    outer = np.outer(std, std)
    outer[outer == 0] = 1e-15
    return cov / outer
