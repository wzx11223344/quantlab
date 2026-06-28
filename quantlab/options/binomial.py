"""
Binomial tree option pricing (Cox-Ross-Rubinstein).

Implements the CRR binomial tree for European and American options,
including Greeks via finite difference on the tree price.

References
----------
Cox, J.C., Ross, S.A., and Rubinstein, M. (1979).
"Option Pricing: A Simplified Approach."
Journal of Financial Economics, 7(3), 229-263.
"""

import numpy as np


def binomial_tree(S, K, T, r, sigma, N=100, option_type='call', style='european'):
    r"""
    Price an option using the CRR binomial tree.

    Parameters
    ----------
    S : float
        Current underlying price.
    K : float
        Strike price.
    T : float
        Time to expiration in years.
    r : float
        Risk-free interest rate (annualized, continuous compounding).
    sigma : float
        Volatility (annualized).
    N : int, optional
        Number of time steps (default 100).
    option_type : str, optional
        'call' or 'put' (default 'call').
    style : str, optional
        'european' or 'american' (default 'european').

    Returns
    -------
    float
        Option price.

    Notes
    -----
    CRR parameterization:

    .. math::
        u &= e^{\sigma\sqrt{\Delta t}} \\
        d &= e^{-\sigma\sqrt{\Delta t}} = 1/u \\
        p &= \frac{e^{r\Delta t} - d}{u - d}

    where :math:`\Delta t = T/N`.

    The option price at each node :math:`(i,j)` (time step *i*, state *j*):

    .. math::
        V_{i,j} = e^{-r\Delta t}\left[p V_{i+1,j+1} + (1-p) V_{i+1,j}\right]

    For American options, the early exercise condition is checked:

    .. math::
        V_{i,j} = \max\left(e^{-r\Delta t}[p V_{i+1,j+1} + (1-p) V_{i+1,j}],
        \text{Intrinsic}_{i,j}\right)

    Examples
    --------
    >>> binomial_tree(100, 105, 1.0, 0.05, 0.2, N=100, option_type='call')
    8.0213...
    >>> binomial_tree(100, 105, 1.0, 0.05, 0.2, N=100, option_type='put', style='american')
    8.3149...
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    # Payoff function
    if option_type == 'call':
        payoff = lambda s: max(s - K, 0.0)
    elif option_type == 'put':
        payoff = lambda s: max(K - s, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Terminal stock prices
    S_final = np.array([S * (u ** j) * (d ** (N - j)) for j in range(N + 1)])

    # Terminal option values
    V = np.array([payoff(s) for s in S_final])

    # Backward induction
    for i in range(N - 1, -1, -1):
        V = discount * (p * V[1:] + (1.0 - p) * V[:-1])

        if style == 'american':
            # Early exercise check
            S_level = np.array([S * (u ** j) * (d ** (i - j)) for j in range(i + 1)])
            exercise = np.array([payoff(s) for s in S_level])
            V = np.maximum(V, exercise)

    return float(V[0])


def binomial_greeks(S, K, T, r, sigma, N=100, option_type='call', style='european', ds=0.01, dt_days=1.0/365.0, dr=0.0001, dsigma=0.0001):
    r"""
    Compute binomial tree Greeks via finite difference.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Standard option parameters.
    N : int, optional
        Time steps (default 100).
    option_type : str, optional
        'call' or 'put'.
    style : str, optional
        'european' or 'american'.
    ds : float, optional
        Price bump for delta/gamma (default 0.01).
    dt_days : float, optional
        Time bump in years for theta (default 1/365).
    dr : float, optional
        Rate bump for rho (default 0.0001).
    dsigma : float, optional
        Vol bump for vega (default 0.0001).

    Returns
    -------
    dict
        Dictionary of Greeks.
    """
    price = binomial_tree(S, K, T, r, sigma, N, option_type, style)

    # Delta and Gamma (central difference)
    price_up = binomial_tree(S + ds, K, T, r, sigma, N, option_type, style)
    price_down = binomial_tree(S - ds, K, T, r, sigma, N, option_type, style)
    delta = (price_up - price_down) / (2.0 * ds)
    gamma = (price_up - 2.0 * price + price_down) / (ds ** 2)

    # Theta (forward difference in time)
    if T > dt_days:
        price_t = binomial_tree(S, K, T - dt_days, r, sigma, N, option_type, style)
        theta = (price_t - price) / dt_days
    else:
        theta = 0.0

    # Vega (central difference in vol)
    price_vup = binomial_tree(S, K, T, r, sigma + dsigma, N, option_type, style)
    price_vdown = binomial_tree(S, K, T, r, sigma - dsigma, N, option_type, style)
    vega = (price_vup - price_vdown) / (2.0 * dsigma)

    # Rho (central difference in rate)
    price_rup = binomial_tree(S, K, T, r + dr, sigma, N, option_type, style)
    price_rdown = binomial_tree(S, K, T, r - dr, sigma, N, option_type, style)
    rho = (price_rup - price_rdown) / (2.0 * dr)

    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho,
    }
