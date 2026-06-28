"""
Yield curve analytics.

Computes implied forward rates, Macaulay duration, modified duration,
convexity, and key rate durations from yield curve data.

References
----------
Fabozzi, F.J. (2016). "Bond Markets, Analysis, and Strategies." 9th ed.
Hull, J.C. (2022). "Options, Futures, and Other Derivatives." 11th ed.
"""

import numpy as np


def forward_rate(spot_rates, maturities, t1, t2, compounding_freq=2):
    r"""
    Compute the implied forward rate between two future dates.

    Parameters
    ----------
    spot_rates : array_like
        Spot rates at known maturities.
    maturities : array_like
        Maturities corresponding to spot rates.
    t1 : float
        Start of forward period in years.
    t2 : float
        End of forward period in years (t2 > t1).
    compounding_freq : int, optional
        Compounding frequency per year (default 2).

    Returns
    -------
    float
        Forward rate for the period [t1, t2].

    Notes
    -----
    The forward rate :math:`f(t_1, t_2)` satisfies:

    .. math::
        (1 + r_{t_2}/f)^{f t_2} = (1 + r_{t_1}/f)^{f t_1}
        \cdot (1 + f(t_1, t_2)/f)^{f(t_2 - t_1)}

    Solving:

    .. math::
        f(t_1, t_2) = f \left[
        \left(\frac{(1 + r_{t_2}/f)^{t_2}}
        {(1 + r_{t_1}/f)^{t_1}}\right)^{1/(t_2 - t_1)} - 1
        \right]
    """
    spot_rates = np.asarray(spot_rates, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    f = compounding_freq

    if t2 <= t1:
        raise ValueError("t2 must be greater than t1")

    # Interpolate spot rates at t1 and t2
    from .bootstrap import _interpolate_spot
    r1 = _interpolate_spot(maturities, spot_rates, t1)
    r2 = _interpolate_spot(maturities, spot_rates, t2)

    df1 = 1.0 / (1.0 + r1 / f) ** (f * t1)
    df2 = 1.0 / (1.0 + r2 / f) ** (f * t2)

    # Forward: df1 = df2 * (1 + forward/f)^(f*(t2-t1))
    forward = f * ((df2 / df1) ** (1.0 / (f * (t2 - t1))) - 1.0)

    return forward


def macaulay_duration(cashflows, times, ytm):
    r"""
    Compute Macaulay duration.

    Parameters
    ----------
    cashflows : array_like
        Cash flow amounts (including final principal repayment).
    times : array_like
        Times to each cash flow in years.
    ytm : float
        Yield to maturity (as decimal, e.g., 0.05).

    Returns
    -------
    float
        Macaulay duration in years.

    Notes
    -----
    Macaulay duration is the weighted average time to receipt of cash flows:

    .. math::
        D_{Mac} = \frac{\sum_{t} t \cdot PV(CF_t)}{\sum_{t} PV(CF_t)}

    where :math:`PV(CF_t) = CF_t / (1 + ytm)^t`.

    This is a measure of interest rate sensitivity: the weighted average
    time until bond cash flows are received.
    """
    cashflows = np.asarray(cashflows, dtype=float)
    times = np.asarray(times, dtype=float)

    pv = cashflows / (1.0 + ytm) ** times
    pv_total = np.sum(pv)

    if pv_total == 0:
        return 0.0

    weighted_times = np.sum(times * pv)

    return weighted_times / pv_total


def modified_duration(mac_dur, ytm, frequency=1):
    r"""
    Compute modified duration from Macaulay duration.

    Parameters
    ----------
    mac_dur : float
        Macaulay duration in years.
    ytm : float
        Yield to maturity (as decimal).
    frequency : int, optional
        Coupon payment frequency per year (default 1).

    Returns
    -------
    float
        Modified duration.

    Notes
    -----
    Modified duration measures the percentage price change for a 1%
    change in yield:

    .. math::
        D_{Mod} = \frac{D_{Mac}}{1 + ytm/f}

    For continuous compounding (frequency = large):

    .. math::
        D_{Mod} = D_{Mac}

    Modified duration is a first-order approximation of price sensitivity:

    .. math::
        \frac{\Delta P}{P} \approx -D_{Mod} \cdot \Delta y

    Examples
    --------
    >>> modified_duration(7.5, 0.05, frequency=2)
    7.317...
    """
    if frequency > 10000:
        return mac_dur
    return mac_dur / (1.0 + ytm / frequency)


def convexity(cashflows, times, ytm):
    r"""
    Compute bond convexity.

    Parameters
    ----------
    cashflows : array_like
        Cash flow amounts.
    times : array_like
        Times to each cash flow in years.
    ytm : float
        Yield to maturity (as decimal).

    Returns
    -------
    float
        Convexity measure.

    Notes
    -----
    Convexity is the second-order sensitivity of bond price to yield:

    .. math::
        C = \frac{1}{P} \frac{d^2 P}{dy^2}
        = \frac{\sum_t t(t+1) \cdot PV(CF_t) / (1+ytm)^2}{\sum_t PV(CF_t)}

    The second-order price approximation with convexity:

    .. math::
        \frac{\Delta P}{P} \approx -D_{Mod} \cdot \Delta y
        + \frac{1}{2} C \cdot (\Delta y)^2

    Examples
    --------
    >>> cf = [5, 5, 5, 5, 105]  # 5% coupon, 5-year bond
    >>> t = [1, 2, 3, 4, 5]
    >>> convexity(cf, t, 0.05)
    23.15...
    """
    cashflows = np.asarray(cashflows, dtype=float)
    times = np.asarray(times, dtype=float)

    pv = cashflows / (1.0 + ytm) ** times
    pv_total = np.sum(pv)

    if pv_total == 0:
        return 0.0

    # Convexity = sum[t*(t+1)*PV] / [P * (1+ytm)^2]
    conv = np.sum(times * (times + 1.0) * pv) / (pv_total * (1.0 + ytm) ** 2)

    return conv


def effective_duration(cashflows, times, ytm, dy=0.0001):
    r"""
    Compute effective duration via finite difference.

    Parameters
    ----------
    cashflows : array_like
        Cash flow amounts.
    times : array_like
        Times to each cash flow.
    ytm : float
        Current yield to maturity.
    dy : float, optional
        Yield shock size (default 0.0001 = 1 bp).

    Returns
    -------
    float
        Effective duration.

    Notes
    -----
    Effective duration is computed numerically:

    .. math::
        D_{Eff} = \frac{P(y - \Delta y) - P(y + \Delta y)}
        {2 \cdot P(y) \cdot \Delta y}
    """
    cashflows = np.asarray(cashflows, dtype=float)
    times = np.asarray(times, dtype=float)

    # Price at current yield
    pv = cashflows / (1.0 + ytm) ** times
    P = np.sum(pv)

    if P == 0:
        return 0.0

    # Price at shifted yields
    pv_down = cashflows / (1.0 + ytm - dy) ** times
    P_down = np.sum(pv_down)

    pv_up = cashflows / (1.0 + ytm + dy) ** times
    P_up = np.sum(pv_up)

    return (P_down - P_up) / (2.0 * P * dy)


def key_rate_duration(cashflows, times, spot_rates, maturities,
                       key_maturities=None, dy=0.0001):
    r"""
    Compute key rate durations (KRD).

    Parameters
    ----------
    cashflows : array_like
        Bond cash flows.
    times : array_like
        Cash flow times in years.
    spot_rates : array_like
        Spot rates at each maturity point.
    maturities : array_like
        Maturities corresponding to spot rates.
    key_maturities : array_like, optional
        Key rate maturities (default: [1, 2, 3, 5, 7, 10, 30]).
    dy : float, optional
        Rate shock size (default 0.0001).

    Returns
    -------
    dict
        Dictionary mapping key maturity to key rate duration.

    Notes
    -----
    Key rate durations measure sensitivity to yield changes at specific
    points on the curve (Ho 1992). Each key rate is shocked individually
    and the impact on bond price is measured.

    .. math::
        KRD_k = \frac{P(y_k - \Delta y) - P(y_k + \Delta y)}
        {2 \cdot P \cdot \Delta y}

    The sum of all KRDs equals the effective duration.
    """
    if key_maturities is None:
        key_maturities = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 30.0])

    cashflows = np.asarray(cashflows, dtype=float)
    times = np.asarray(times, dtype=float)
    spot_rates = np.asarray(spot_rates, dtype=float)
    maturities = np.asarray(maturities, dtype=float)

    from .bootstrap import _interpolate_spot

    # Base price
    base_price = 0.0
    for cf, t in zip(cashflows, times):
        r = _interpolate_spot(maturities, spot_rates, t)
        base_price += cf / (1.0 + r) ** t

    if base_price == 0:
        return {k: 0.0 for k in key_maturities}

    # Compute KRD for each key maturity
    krd = {}
    for kt in key_maturities:
        # Shock this key rate
        spot_up = spot_rates.copy()
        spot_down = spot_rates.copy()

        # Find the closest maturity index
        idx = np.argmin(np.abs(maturities - kt))
        spot_up[idx] += dy
        spot_down[idx] -= dy

        price_up = 0.0
        price_down = 0.0
        for cf, t in zip(cashflows, times):
            r_up = _interpolate_spot(maturities, spot_up, t)
            r_down = _interpolate_spot(maturities, spot_down, t)
            price_up += cf / (1.0 + r_up) ** t
            price_down += cf / (1.0 + r_down) ** t

        krd[kt] = (price_down - price_up) / (2.0 * base_price * dy) / 100.0

    return krd
