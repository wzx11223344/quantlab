"""
Risk and performance metrics.

Computes standard financial risk-adjusted return measures including
Sharpe ratio, Sortino ratio, maximum drawdown, Calmar ratio, and
information ratio.

References
----------
Sharpe, W.F. (1966). "Mutual Fund Performance."
Sortino, F.A. and van der Meer, R. (1991). "Downside Risk."
Young, T.W. (1991). "Calmar Ratio: A Smoother Tool."
"""

import numpy as np


def sharpe_ratio(returns, rf=0.0, periods_per_year=252):
    r"""
    Compute the annualized Sharpe ratio.

    Parameters
    ----------
    returns : array_like
        Array of periodic returns.
    rf : float, optional
        Risk-free rate (annualized, default 0).
    periods_per_year : int, optional
        Number of periods per year (252 for daily, 12 for monthly, default 252).

    Returns
    -------
    float
        Annualized Sharpe ratio.

    Notes
    -----
    The Sharpe ratio measures excess return per unit of risk:

    .. math::
        SR = \frac{\bar{R}_p - r_f}{\sigma_p}

    where :math:`\bar{R}_p` is the annualized mean return and :math:`\sigma_p`
    is the annualized standard deviation. If :math:`\sigma_p = 0`, returns 0.

    Examples
    --------
    >>> returns = np.array([0.001, -0.002, 0.003, 0.001, 0.0])
    >>> sharpe_ratio(returns, rf=0.02, periods_per_year=252)
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 2:
        return 0.0

    excess = returns - rf / periods_per_year
    mean_excess = np.mean(excess)
    std_ret = np.std(returns, ddof=1)

    if std_ret == 0:
        return 0.0

    return np.sqrt(periods_per_year) * mean_excess / std_ret


def sortino_ratio(returns, rf=0.0, target=0.0, periods_per_year=252):
    r"""
    Compute the annualized Sortino ratio.

    Parameters
    ----------
    returns : array_like
        Array of periodic returns.
    rf : float, optional
        Risk-free rate (annualized, default 0).
    target : float, optional
        Minimum acceptable return (MAR) per period (default 0).
    periods_per_year : int, optional
        Number of periods per year (default 252).

    Returns
    -------
    float
        Annualized Sortino ratio.

    Notes
    -----
    The Sortino ratio uses downside deviation instead of total volatility:

    .. math::
        \text{Sortino} = \frac{\bar{R}_p - r_f}{\sigma_d}

    where the downside deviation is:

    .. math::
        \sigma_d = \sqrt{\frac{1}{n}\sum_{t=1}^{n}
        \min(R_t - \text{MAR}, 0)^2}

    Both numerator and denominator are annualized via :math:`\sqrt{P}`.
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 2:
        return 0.0

    excess = returns - rf / periods_per_year
    mean_excess = np.mean(excess)

    # Downside deviation
    downside = np.minimum(returns - target, 0.0)
    downside_std = np.sqrt(np.mean(downside ** 2))

    if downside_std == 0:
        return 0.0

    return np.sqrt(periods_per_year) * mean_excess / downside_std


def max_drawdown(prices):
    r"""
    Compute the maximum drawdown from a price series.

    Parameters
    ----------
    prices : array_like
        Array of prices (e.g., portfolio values over time).

    Returns
    -------
    float
        Maximum drawdown as a positive fraction (e.g., 0.25 means 25%).

    Notes
    -----
    Maximum drawdown is the largest peak-to-trough decline:

    .. math::
        \text{MDD} = \max_{t} \left( \frac{\max_{s \le t} P_s - P_t}
        {\max_{s \le t} P_s} \right)

    This is always reported as a positive number.

    Examples
    --------
    >>> prices = np.array([100, 105, 95, 110, 90, 100])
    >>> max_drawdown(prices)
    0.1818...  # 18.18%
    """
    prices = np.asarray(prices, dtype=float)
    if len(prices) < 2:
        return 0.0

    running_max = np.maximum.accumulate(prices)
    drawdowns = (running_max - prices) / running_max
    return float(np.max(drawdowns))


def calmar_ratio(returns, periods_per_year=252):
    r"""
    Compute the Calmar ratio (annualized return / max drawdown).

    Parameters
    ----------
    returns : array_like
        Array of periodic returns.
    periods_per_year : int, optional
        Number of periods per year (default 252).

    Returns
    -------
    float
        Calmar ratio.

    Notes
    -----
    The Calmar ratio is:

    .. math::
        \text{Calmar} = \frac{\text{Annualized Return}}{\text{Maximum Drawdown}}

    Higher values indicate better risk-adjusted returns. If max drawdown
    is zero, returns the annualized return directly.
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 2:
        return 0.0

    # Reconstruct price series from returns
    prices = np.cumprod(1.0 + returns)

    # Annualized return
    n = len(returns)
    total_return = prices[-1]
    annual_return = total_return ** (periods_per_year / n) - 1.0

    mdd = max_drawdown(prices)
    if mdd == 0:
        return annual_return

    return annual_return / mdd


def information_ratio(returns, benchmark_returns, periods_per_year=252):
    r"""
    Compute the annualized Information Ratio.

    Parameters
    ----------
    returns : array_like
        Array of portfolio returns.
    benchmark_returns : array_like
        Array of benchmark returns.
    periods_per_year : int, optional
        Number of periods per year (default 252).

    Returns
    -------
    float
        Annualized Information Ratio.

    Notes
    -----
    The Information Ratio measures excess return over a benchmark per unit
    of tracking error:

    .. math::
        IR = \frac{\bar{R}_p - \bar{R}_b}{\sigma(R_p - R_b)}
        \cdot \sqrt{P}

    where :math:`\bar{R}_p` is the mean portfolio return, :math:`\bar{R}_b`
    is the mean benchmark return, and :math:`\sigma(R_p - R_b)` is the
    tracking error.
    """
    returns = np.asarray(returns, dtype=float)
    benchmark_returns = np.asarray(benchmark_returns, dtype=float)

    if len(returns) != len(benchmark_returns):
        raise ValueError("returns and benchmark_returns must have the same length")
    if len(returns) < 2:
        return 0.0

    excess = returns - benchmark_returns
    mean_excess = np.mean(excess)
    std_excess = np.std(excess, ddof=1)

    if std_excess == 0:
        return 0.0

    return np.sqrt(periods_per_year) * mean_excess / std_excess


def tracking_error(returns, benchmark_returns, periods_per_year=252):
    r"""
    Compute annualized tracking error.

    Parameters
    ----------
    returns : array_like
        Portfolio returns.
    benchmark_returns : array_like
        Benchmark returns.
    periods_per_year : int, optional
        Periods per year (default 252).

    Returns
    -------
    float
        Annualized tracking error.

    Notes
    -----
    :math:`\text{TE} = \sigma(R_p - R_b) \cdot \sqrt{P}`
    """
    returns = np.asarray(returns, dtype=float)
    benchmark_returns = np.asarray(benchmark_returns, dtype=float)
    excess = returns - benchmark_returns
    return np.std(excess, ddof=1) * np.sqrt(periods_per_year)


def omega_ratio(returns, threshold=0.0):
    r"""
    Compute the Omega ratio.

    Parameters
    ----------
    returns : array_like
        Array of returns.
    threshold : float, optional
        Return threshold (default 0, i.e., risk-free rate).

    Returns
    -------
    float
        Omega ratio.

    Notes
    -----
    .. math::
        \Omega(L) = \frac{\int_L^\infty [1 - F(x)]dx}{\int_{-\infty}^L F(x)dx}

    where :math:`F` is the CDF of returns. Roughly, it is the ratio of
    gains above threshold to losses below threshold.
    """
    returns = np.asarray(returns, dtype=float)
    gains = np.sum(np.maximum(returns - threshold, 0.0))
    losses = np.sum(np.maximum(threshold - returns, 0.0))
    if losses == 0:
        return np.inf if gains > 0 else 1.0
    return gains / losses
