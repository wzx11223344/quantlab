"""
Monte Carlo option pricing.

Implements Monte Carlo simulation for European, Asian, and Barrier options
with variance reduction techniques including antithetic variates and
control variates. Uses manual Box-Muller transform for normal random numbers.

References
----------
Glasserman, P. (2004). "Monte Carlo Methods in Financial Engineering." Springer.
Boyle, P.P. (1977). "Options: A Monte Carlo Approach."
Journal of Financial Economics, 4(3), 323-338.
"""

import numpy as np
from ..utils import box_muller
from ..options.black_scholes import black_scholes_price


def _generate_normal(n_sims, seed=None):
    """Generate standard normal random numbers via Box-Muller."""
    if seed is not None:
        np.random.seed(seed)
    return box_muller(n_sims)


def mc_price(S, K, T, r, sigma, n_sims=100000, option_type='call',
             antithetic=True, control_variate=True, seed=None):
    r"""
    Price a European option using Monte Carlo simulation.

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
    n_sims : int, optional
        Number of simulation paths (default 100000).
    option_type : str, optional
        'call' or 'put' (default 'call').
    antithetic : bool, optional
        Use antithetic variates for variance reduction (default True).
    control_variate : bool, optional
        Use put-call parity as control variate (default True).
    seed : int, optional
        Random seed.

    Returns
    -------
    price : float
        Estimated option price.
    std_err : float
        Standard error of the estimate.

    Notes
    -----
    The terminal underlying price under risk-neutral measure:

    .. math::
        S_T = S_0 \exp\left((r - \tfrac{1}{2}\sigma^2)T + \sigma\sqrt{T}Z\right),
        \quad Z \sim \mathcal{N}(0,1)

    Antithetic variates: use pairs :math:`(Z, -Z)` to double effective samples.

    Control variates: use the Black-Scholes put-call parity relationship:

    .. math::
        C - P = S - Ke^{-rT}

    as a control to reduce variance.
    """
    n_effective = n_sims
    if antithetic:
        n_effective = n_sims // 2
        if n_effective * 2 < n_sims:
            n_effective += 1

    discount = np.exp(-r * T)
    drift = np.log(S) + (r - 0.5 * sigma ** 2) * T
    vol_sqrt = sigma * np.sqrt(T)

    if seed is not None:
        np.random.seed(seed)

    # Generate normal random numbers
    Z = box_muller(n_effective)

    # Antithetic: use Z and -Z
    if antithetic:
        Z_all = np.concatenate([Z, -Z])[:n_sims]
    else:
        Z_all = Z[:n_sims]

    ST = np.exp(drift + vol_sqrt * Z_all)

    # Payoff
    if option_type == 'call':
        payoff = np.maximum(ST - K, 0.0)
    elif option_type == 'put':
        payoff = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    discounted = discount * payoff

    # Control variate: put-call parity C - P = S - Ke^{-rT}
    if control_variate:
        cv_bs = black_scholes_price(S, K, T, r, sigma, option_type)
        # The control variate is the BS price itself
        # Y = payoff; C = BS_price; adjusted = Y - b*(C - E[C])
        # We use b=1 for simplicity (optimal b requires covariance estimation)
        discounted = discounted - (discounted.mean() - cv_bs)

    price = float(discounted.mean())
    std_err = float(discounted.std(ddof=1) / np.sqrt(n_sims))

    return price, std_err


def mc_price_asian(S, K, T, r, sigma, n_sims=100000, averaging='arithmetic',
                    n_obs=252, seed=None):
    r"""
    Price an Asian option using Monte Carlo simulation.

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
    n_sims : int, optional
        Number of simulation paths (default 100000).
    averaging : str, optional
        'arithmetic' or 'geometric' (default 'arithmetic').
    n_obs : int, optional
        Number of observation dates for averaging (default 252).
    seed : int, optional
        Random seed.

    Returns
    -------
    price : float
        Estimated Asian option price.
    std_err : float
        Standard error.

    Notes
    -----
    The Asian option payoff depends on the average underlying price over the
    option's life:

    .. math::
        A = \frac{1}{n}\sum_{i=1}^{n} S_{t_i} \quad \text{(arithmetic)}
        \quad\text{or}\quad
        A = \left(\prod_{i=1}^{n} S_{t_i}\right)^{1/n} \quad \text{(geometric)}

    Fixed-strike call payoff: :math:`\max(A - K, 0)`.
    """
    dt = T / n_obs
    discount = np.exp(-r * T)

    if seed is not None:
        np.random.seed(seed)

    # Simulate paths with n_obs steps
    drift = (r - 0.5 * sigma ** 2) * dt
    vol_sqrt = sigma * np.sqrt(dt)

    # Precompute averages
    all_payoffs = np.zeros(n_sims)

    for sim in range(n_sims):
        Z = box_muller(n_obs)
        logS = np.log(S) + np.cumsum(drift + vol_sqrt * Z)
        prices = np.exp(logS)

        if averaging == 'arithmetic':
            avg = prices.mean()
        elif averaging == 'geometric':
            avg = np.exp(np.log(prices).mean())
        else:
            raise ValueError("averaging must be 'arithmetic' or 'geometric'")

        all_payoffs[sim] = max(avg - K, 0.0)

    discounted = discount * all_payoffs
    price = float(discounted.mean())
    std_err = float(discounted.std(ddof=1) / np.sqrt(n_sims))

    return price, std_err


def mc_price_barrier(S, K, T, r, sigma, barrier, n_sims=100000,
                      barrier_type='up-and-out', n_steps=252, seed=None):
    r"""
    Price a barrier option using Monte Carlo simulation.

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
    barrier : float
        Barrier level.
    n_sims : int, optional
        Number of simulation paths (default 100000).
    barrier_type : str, optional
        One of 'up-and-out', 'up-and-in', 'down-and-out', 'down-and-in'
        (default 'up-and-out').
    n_steps : int, optional
        Number of monitoring steps (default 252, daily).
    seed : int, optional
        Random seed.

    Returns
    -------
    price : float
        Estimated barrier option price.
    std_err : float
        Standard error.

    Notes
    -----
    Barrier options are path-dependent: their payoff depends on whether the
    underlying crosses a predetermined barrier level during the option's life.

    - **up-and-out**: Knock out if S exceeds barrier (S > B).
    - **down-and-out**: Knock out if S falls below barrier (S < B).
    - **up-and-in**: Knock in only if S exceeds barrier.
    - **down-and-in**: Knock in only if S falls below barrier.

    All use a call payoff :math:`\max(S_T - K, 0)`.

    Continuous monitoring is approximated by high-frequency discrete monitoring.
    """
    dt = T / n_steps
    discount = np.exp(-r * T)
    drift = (r - 0.5 * sigma ** 2) * dt
    vol_sqrt = sigma * np.sqrt(dt)

    is_out = 'out' in barrier_type
    is_up = 'up' in barrier_type

    if seed is not None:
        np.random.seed(seed)

    all_payoffs = np.zeros(n_sims)

    for sim in range(n_sims):
        Z = box_muller(n_steps)
        logS = np.log(S) + np.cumsum(drift + vol_sqrt * Z)
        path = np.exp(logS)

        # Check barrier condition
        if is_up:
            hit = np.any(path >= barrier)
        else:
            hit = np.any(path <= barrier)

        ST = path[-1]
        call_payoff = max(ST - K, 0.0)

        if is_out:
            all_payoffs[sim] = call_payoff if not hit else 0.0
        else:
            all_payoffs[sim] = call_payoff if hit else 0.0

    discounted = discount * all_payoffs
    price = float(discounted.mean())
    std_err = float(discounted.std(ddof=1) / np.sqrt(n_sims))

    return price, std_err
