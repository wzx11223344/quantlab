"""
Realized and range-based volatility estimators.

Implements standard volatility estimators from high-frequency and
OHLC (Open-High-Low-Close) price data, including Parkinson,
Garman-Klass, and Yang-Zhang estimators.

References
----------
Parkinson, M. (1980). "The Extreme Value Method for Estimating the
Variance of the Rate of Return." Journal of Business, 53(1), 61-65.
Garman, M.B. and Klass, M.J. (1980). "On the Estimation of Security
Price Volatilities from Historical Data." Journal of Business, 53(1), 67-78.
Yang, D. and Zhang, Q. (2000). "Drift-Independent Volatility Estimation
Based on High, Low, Open, and Close Prices." Journal of Business, 73(3), 477-491.
"""

import numpy as np


def realized_volatility(high, low, open_p, close):
    r"""
    Compute realized volatility from intraday OHLC data.

    Parameters
    ----------
    high : array_like
        Array of high prices.
    low : array_like
        Array of low prices.
    open_p : array_like
        Array of open prices.
    close : array_like
        Array of close prices.

    Returns
    -------
    float
        Realized volatility (standard deviation of log returns).

    Notes
    -----
    Realized volatility is computed as:

    .. math::
        RV = \sqrt{\sum_{t=1}^T r_t^2}

    where :math:`r_t = \log(C_t / C_{t-1})` is the close-to-close log return.
    This is the most basic realized measure; for high-frequency data,
    more sophisticated estimators using tick data are preferred.
    """
    close = np.asarray(close, dtype=float)
    if len(close) < 2:
        return 0.0
    log_rets = np.log(close[1:] / close[:-1])
    return float(np.std(log_rets, ddof=1))


def parkinson_volatility(high, low, periods_per_year=252):
    r"""
    Compute Parkinson (1980) range-based volatility estimator.

    Parameters
    ----------
    high : array_like
        Array of high prices.
    low : array_like
        Array of low prices.
    periods_per_year : int, optional
        Annualization factor (default 252 for daily).

    Returns
    -------
    float
        Annualized Parkinson volatility.

    Notes
    -----
    The Parkinson estimator uses only high and low prices:

    .. math::
        \sigma_P^2 = \frac{1}{4n\ln 2} \sum_{t=1}^n
        \left(\ln\frac{H_t}{L_t}\right)^2

    This estimator is 5.2x more efficient than the close-to-close estimator
    under the assumption of driftless geometric Brownian motion.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)

    n = len(high)
    if n < 1:
        return 0.0

    log_hl = np.log(high / low)
    variance = np.sum(log_hl ** 2) / (4.0 * n * np.log(2.0))

    return np.sqrt(variance * periods_per_year)


def garman_klass_volatility(open_p, high, low, close, periods_per_year=252):
    r"""
    Compute Garman-Klass (1980) OHLC volatility estimator.

    Parameters
    ----------
    open_p : array_like
        Array of open prices.
    high : array_like
        Array of high prices.
    low : array_like
        Array of low prices.
    close : array_like
        Array of close prices.
    periods_per_year : int, optional
        Annualization factor (default 252).

    Returns
    -------
    float
        Annualized Garman-Klass volatility.

    Notes
    -----
    The Garman-Klass estimator incorporates open, high, low, and close:

    .. math::
        \sigma_{GK}^2 = \frac{1}{n}\sum_{t=1}^n \left[
        \frac{1}{2}\left(\ln\frac{H_t}{L_t}\right)^2
        - (2\ln 2 - 1)\left(\ln\frac{C_t}{O_t}\right)^2
        \right]

    This is approximately 7.4x more efficient than close-to-close.
    """
    open_p = np.asarray(open_p, dtype=float)
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)

    n = len(open_p)
    if n < 1:
        return 0.0

    log_ho = np.log(high / open_p)
    log_lo = np.log(low / open_p)
    log_co = np.log(close / open_p)

    variance = np.mean(
        0.5 * (log_ho - log_lo) ** 2
        - (2.0 * np.log(2.0) - 1.0) * log_co ** 2
    )

    variance = max(variance, 0.0)
    return np.sqrt(variance * periods_per_year)


def yang_zhang_volatility(open_p, high, low, close, periods_per_year=252):
    r"""
    Compute Yang-Zhang (2000) drift-independent volatility estimator.

    This is the most sophisticated OHLC estimator, unbiased under both
    drift and opening jumps (gaps between close and next open).

    Parameters
    ----------
    open_p : array_like
        Array of open prices.
    high : array_like
        Array of high prices.
    low : array_like
        Array of low prices.
    close : array_like
        Array of close prices.
    periods_per_year : int, optional
        Annualization factor (default 252).

    Returns
    -------
    float
        Annualized Yang-Zhang volatility.

    Notes
    -----
    The Yang-Zhang estimator combines overnight and intraday components:

    .. math::
        \sigma_{YZ}^2 = \sigma_o^2 + k\sigma_c^2 + (1-k)\sigma_{RS}^2

    where:

    .. math::
        \sigma_o^2 &= \frac{1}{n-1}\sum(\ln(O_t/C_{t-1}) - \overline{o})^2 \\
        \sigma_c^2 &= \frac{1}{n-1}\sum(\ln(C_t/O_t) - \overline{c})^2 \\
        \sigma_{RS}^2 &= \frac{1}{n}\sum\left[
        \ln\frac{H_t}{C_t}\ln\frac{H_t}{O_t}
        + \ln\frac{L_t}{C_t}\ln\frac{L_t}{O_t}
        \right] \\
        k &= \frac{0.34}{1.34 + (n+1)/(n-1)}

    This estimator is approximately 7.4x more efficient than close-to-close
    and is unbiased in the presence of drift.
    """
    open_p = np.asarray(open_p, dtype=float)
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)

    n = len(open_p)
    if n < 2:
        return 0.0

    # Overnight (close-to-open) log returns
    log_open_close = np.log(open_p[1:] / close[:-1])
    var_o = np.var(log_open_close, ddof=1)

    # Intraday (open-to-close) log returns
    log_close_open = np.log(close / open_p)
    var_c = np.var(log_close_open, ddof=1)

    # Rogers-Satchell component
    log_ho = np.log(high / open_p)
    log_lo = np.log(low / open_p)
    log_hc = np.log(high / close)
    log_lc = np.log(low / close)

    rs = np.mean(log_ho * log_hc + log_lo * log_lc)

    # Weighting factor k
    k = 0.34 / (1.34 + (n + 1) / (n - 1))

    variance = var_o + k * var_c + (1.0 - k) * rs
    variance = max(variance, 0.0)

    return np.sqrt(variance * periods_per_year)


def close_to_close_volatility(close, periods_per_year=252):
    r"""
    Compute the classic close-to-close volatility estimator.

    Parameters
    ----------
    close : array_like
        Array of close prices.
    periods_per_year : int, optional
        Annualization factor (default 252).

    Returns
    -------
    float
        Annualized close-to-close volatility.

    Notes
    -----
    .. math::
        \sigma = \sqrt{\frac{1}{n-1}\sum_{t=2}^n (r_t - \bar{r})^2}
        \cdot \sqrt{P}

    where :math:`r_t = \ln(C_t / C_{t-1})` and :math:`P` is the annualization factor.
    """
    close = np.asarray(close, dtype=float)
    if len(close) < 2:
        return 0.0
    log_rets = np.log(close[1:] / close[:-1])
    return float(np.std(log_rets, ddof=1) * np.sqrt(periods_per_year))
