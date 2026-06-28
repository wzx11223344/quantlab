"""
Black-Scholes option pricing model.

Implements the exact Black-Scholes-Merton formula for European options,
all five first-order Greeks, and implied volatility via Newton-Raphson.

All special functions (norm.cdf, norm.pdf) are computed manually using
the error function approximation to avoid scipy dependency.

References
----------
Black, F. and Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities."
Journal of Political Economy, 81(3), 637-654.
Merton, R.C. (1973). "Theory of Rational Option Pricing."
Bell Journal of Economics and Management Science, 4(1), 141-183.
"""

import numpy as np
from ..utils import norm_cdf, norm_pdf


def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    r"""
    Price a European option using the Black-Scholes formula.

    Parameters
    ----------
    S : float or array_like
        Current underlying price.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Risk-free interest rate (annualized, continuous compounding).
    sigma : float
        Volatility of the underlying (annualized).
    option_type : str, optional
        'call' or 'put' (default 'call').

    Returns
    -------
    float or ndarray
        Option price.

    Notes
    -----
    The Black-Scholes formula for a European call option is:

    .. math::
        C &= S \Phi(d_1) - K e^{-rT} \Phi(d_2) \\
        P &= K e^{-rT} \Phi(-d_2) - S \Phi(-d_1) \\
        d_1 &= \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}} \\
        d_2 &= d_1 - \sigma\sqrt{T}

    where :math:`\Phi` is the standard normal CDF.

    Examples
    --------
    >>> black_scholes_price(100, 105, 1.0, 0.05, 0.2, 'call')
    8.0213...
    >>> black_scholes_price(100, 105, 1.0, 0.05, 0.2, 'put')
    7.9004...
    """
    S = np.asarray(S, dtype=float)
    if sigma <= 0 or T <= 0:
        if option_type == 'call':
            return np.maximum(S - K * np.exp(-r * T), 0.0)
        else:
            return np.maximum(K * np.exp(-r * T) - S, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return float(price) if np.isscalar(S) else price


def _compute_d1_d2(S, K, T, r, sigma):
    """Compute d1 and d2 for Black-Scholes."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes_greeks(S, K, T, r, sigma):
    r"""
    Compute all five first-order Black-Scholes Greeks.

    Parameters
    ----------
    S : float
        Current underlying price.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Risk-free interest rate (annualized).
    sigma : float
        Volatility (annualized).

    Returns
    -------
    dict
        Dictionary with keys 'delta', 'gamma', 'theta', 'vega', 'rho'.

    Notes
    -----
    The Greeks for a European call option:

    .. math::
        \Delta &= \Phi(d_1) \\
        \Gamma &= \frac{\phi(d_1)}{S\sigma\sqrt{T}} \\
        \Theta &= -\frac{S\phi(d_1)\sigma}{2\sqrt{T}} - rKe^{-rT}\Phi(d_2) \\
        \mathcal{V} &= S\phi(d_1)\sqrt{T} \\
        \rho &= KTe^{-rT}\Phi(d_2)

    where :math:`\phi` is the standard normal PDF.
    Put Greeks follow from put-call parity.
    """
    d1, d2 = _compute_d1_d2(S, K, T, r, sigma)
    pdf_d1 = norm_pdf(d1)
    discount = np.exp(-r * T)

    # Call Greeks
    delta_call = norm_cdf(d1)
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    theta_call = (
        -S * pdf_d1 * sigma / (2.0 * np.sqrt(T))
        - r * K * discount * norm_cdf(d2)
    )
    vega = S * pdf_d1 * np.sqrt(T)
    rho_call = K * T * discount * norm_cdf(d2)

    # Put Greeks via put-call parity
    delta_put = delta_call - 1.0
    theta_put = theta_call + r * K * discount
    rho_put = rho_call - K * T * discount

    return {
        'delta_call': delta_call,
        'delta_put': delta_put,
        'gamma': gamma,
        'theta_call': theta_call,
        'theta_put': theta_put,
        'vega': vega,
        'rho_call': rho_call,
        'rho_put': rho_put,
    }


def implied_volatility(market_price, S, K, T, r, option_type='call', tol=1e-8, max_iter=100):
    r"""
    Compute implied volatility via the Newton-Raphson method.

    Parameters
    ----------
    market_price : float
        Observed market price of the option.
    S : float
        Current underlying price.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Risk-free interest rate (annualized).
    option_type : str, optional
        'call' or 'put' (default 'call').
    tol : float, optional
        Convergence tolerance (default 1e-8).
    max_iter : int, optional
        Maximum iterations (default 100).

    Returns
    -------
    float
        Implied volatility.

    Raises
    ------
    ValueError
        If the root is not found or market price is outside theoretical bounds.

    Notes
    -----
    Uses Newton-Raphson root finding:

    .. math::
        \sigma_{n+1} = \sigma_n - \frac{BS(\sigma_n) - P_{market}}{\mathcal{V}(\sigma_n)}

    where :math:`\mathcal{V}` is vega. The denominator being zero causes failure,
    so vega is clamped to a minimum absolute value.

    The theoretical minimum option price (for calls) is :math:`\max(0, S - Ke^{-rT})`.
    """
    # Check bounds
    intrinsic = max(0.0, S - K * np.exp(-r * T)) if option_type == 'call' else max(0.0, K * np.exp(-r * T) - S)

    if market_price <= intrinsic:
        raise ValueError(
            f"Market price {market_price} is at or below intrinsic value {intrinsic}. "
            "No valid implied volatility."
        )

    # Initial guess using Brenner-Subrahmanyam for ATM
    sigma = np.sqrt(2.0 * np.pi / T) * (market_price / S)

    for i in range(max_iter):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        greeks = black_scholes_greeks(S, K, T, r, sigma)
        vega = greeks['vega']

        if abs(vega) < 1e-15:
            ve = 1e-10 if vega >= 0 else -1e-10

        diff = price - market_price

        if abs(diff) < tol:
            return sigma

        sigma = sigma - diff / vega

        # Ensure sigma stays positive
        if sigma <= 0:
            sigma = 0.001

    raise ValueError(
        f"Implied volatility did not converge after {max_iter} iterations. "
        f"Last estimate: {sigma:.6f}"
    )
