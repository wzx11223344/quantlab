"""
Yield curve bootstrapping.

Constructs a zero-coupon (spot) yield curve from par bond yields
using the bootstrapping methodology. Supports computing discount
factors from spot rates.

References
----------
Hull, J.C. (2022). "Options, Futures, and Other Derivatives." 11th ed.
James, J. and Webber, N. (2000). "Interest Rate Modelling." Wiley.
"""

import numpy as np


def bootstrap_spot_curve(par_rates, maturities, compounding_freq=2):
    r"""
    Bootstrap a zero-coupon (spot) yield curve from par bond yields.

    Parameters
    ----------
    par_rates : array_like
        Par bond yields (as decimals, e.g., 0.05 = 5%) for each maturity.
    maturities : array_like
        Maturities in years corresponding to each par rate.
    compounding_freq : int, optional
        Compounding frequency per year (default 2 for semi-annual).

    Returns
    -------
    spot_rates : ndarray
        Bootstrapped zero-coupon rates at each maturity.
    discount_factors : ndarray
        Corresponding discount factors.

    Notes
    -----
    Bootstrapping assumes par bonds (price = face value). For a par bond
    with coupon rate c and maturity T, the cash flows are discounted:

    .. math::
        1 = \sum_{i=1}^{m} \frac{c/f}{(1 + r_{t_i}/f)^{i}} +
        \frac{1 + c/f}{(1 + r_T/f)^{m}}

    where :math:`f` is the compounding frequency, :math:`m = f \cdot T` is
    the number of periods, and :math:`r_{t_i}` are the spot rates.

    The bootstrapping solves iteratively: given spot rates up to :math:`T_{n-1}`,
    solve for the spot rate at :math:`T_n`.

    Examples
    --------
    >>> par_rates = [0.01, 0.015, 0.02, 0.025, 0.03]
    >>> maturities = [0.5, 1.0, 1.5, 2.0, 3.0]
    >>> spot, df = bootstrap_spot_curve(par_rates, maturities)
    """
    par_rates = np.asarray(par_rates, dtype=float)
    maturities = np.asarray(maturities, dtype=float)

    if len(par_rates) != len(maturities):
        raise ValueError("par_rates and maturities must have the same length")

    n = len(maturities)
    spot_rates = np.zeros(n)
    discount_factors = np.zeros(n)

    for i in range(n):
        T = maturities[i]
        c = par_rates[i]  # Coupon rate (par bond)
        f = compounding_freq
        m = int(round(T * f))  # Number of coupon periods

        if m <= 1:
            # Single payment: spot rate = par rate
            spot_rates[i] = c
            discount_factors[i] = 1.0 / (1.0 + c / f) ** m
            continue

        # Interpolate known spot rates for intermediate cash flow dates
        # Sum of PV of coupon payments before maturity
        pv_coupons = 0.0
        for j in range(1, m):
            t_j = j / f  # Time of j-th cash flow
            # Find the interpolated spot rate at t_j
            r_j = _interpolate_spot(maturities[:i+1], spot_rates[:i+1], t_j)
            pv_coupons += (c / f) / (1.0 + r_j / f) ** (f * t_j)

        # Solve for spot rate: 1 = PV(coupons) + (1 + c/f) / (1 + r_T/f)^m
        pv_remaining = 1.0 - pv_coupons
        if pv_remaining <= 0:
            # Edge case: negative or zero PV remaining
            spot_rates[i] = c * 2.0
            discount_factors[i] = pv_coupons / (1.0 + c / f)
        else:
            # (1 + r/f)^m = (1 + c/f) / pv_remaining
            factor = (1.0 + c / f) / pv_remaining
            r = f * (factor ** (1.0 / m) - 1.0)
            spot_rates[i] = r
            # Discount factor
            discount_factors[i] = 1.0 / (1.0 + r / f) ** m

    return spot_rates, discount_factors


def _interpolate_spot(maturities, spot_rates, t):
    """
    Linear interpolation of spot rates.

    Parameters
    ----------
    maturities : ndarray
        Known maturities.
    spot_rates : ndarray
        Known spot rates.
    t : float
        Target maturity.

    Returns
    -------
    float
        Interpolated spot rate.
    """
    if t <= maturities[0]:
        return spot_rates[0]
    if t >= maturities[-1]:
        return spot_rates[-1]

    # Find bracketing points
    for i in range(len(maturities) - 1):
        if maturities[i] <= t <= maturities[i + 1]:
            # Linear interpolation
            frac = (t - maturities[i]) / (maturities[i + 1] - maturities[i])
            return spot_rates[i] + frac * (spot_rates[i + 1] - spot_rates[i])

    # Fallback: nearest
    idx = np.argmin(np.abs(maturities - t))
    return spot_rates[idx]


def discount_factors_from_spot(spot_rates, maturities, compounding_freq=2):
    r"""
    Compute discount factors from spot rates.

    Parameters
    ----------
    spot_rates : array_like
        Zero-coupon spot rates (as decimals) at each maturity.
    maturities : array_like
        Maturities in years.
    compounding_freq : int, optional
        Compounding frequency per year (default 2).

    Returns
    -------
    ndarray
        Discount factors: :math:`DF(T) = (1 + r_T/f)^{-fT}`.

    Notes
    -----
    For continuous compounding, set compounding_freq to a very large value
    (e.g., 1e9). The discount factor is then:

    .. math::
        DF(T) = e^{-r_T \cdot T}
    """
    spot_rates = np.asarray(spot_rates, dtype=float)
    maturities = np.asarray(maturities, dtype=float)
    f = compounding_freq

    if compounding_freq > 10000:
        # Approximate continuous compounding
        return np.exp(-spot_rates * maturities)

    return 1.0 / (1.0 + spot_rates / f) ** (f * maturities)
