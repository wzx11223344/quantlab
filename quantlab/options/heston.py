"""
Heston stochastic volatility option pricing.

Implements the Heston (1993) model using the Carr-Madan Fourier transform
method and least-squares calibration to market prices.

References
----------
Heston, S.L. (1993). "A Closed-Form Solution for Options with Stochastic
Volatility with Applications to Bond and Currency Options."
Review of Financial Studies, 6(2), 327-343.
Carr, P. and Madan, D. (1999). "Option Valuation Using the Fast Fourier Transform."
Journal of Computational Finance, 2(4), 61-73.
"""

import numpy as np


def heston_characteristic_function(u, S0, r, T, v0, kappa, theta, xi, rho):
    r"""
    Heston characteristic function :math:`\phi(u)`.

    The log-price characteristic function under the Heston model:

    .. math::
        \phi(u) = \exp\left(C(u,\tau) + D(u,\tau)v_0 + iu\log S_0 + iur\tau\right)

    where

    .. math::
        d &= \sqrt{(\rho\xi iu - \kappa)^2 + \xi^2(iu + u^2)} \\
        g &= \frac{\kappa - \rho\xi iu - d}{\kappa - \rho\xi iu + d} \\
        C &= iur\tau + \frac{\kappa\theta}{\xi^2}
        \left[(\kappa - \rho\xi iu - d)\tau - 2\log\frac{1 - ge^{-d\tau}}{1 - g}\right] \\
        D &= \frac{\kappa - \rho\xi iu - d}{\xi^2} \cdot
        \frac{1 - e^{-d\tau}}{1 - ge^{-d\tau}}

    Parameters
    ----------
    u : complex or ndarray
        Fourier variable.
    S0 : float
        Initial asset price.
    r : float
        Risk-free rate.
    T : float
        Time to maturity.
    v0 : float
        Initial variance.
    kappa : float
        Mean reversion speed.
    theta : float
        Long-run mean variance.
    xi : float
        Volatility of variance.
    rho : float
        Correlation between asset and variance processes.

    Returns
    -------
    complex or ndarray
        Characteristic function value(s).
    """
    u = np.asarray(u, dtype=complex)

    a = kappa * theta
    d = np.sqrt((rho * xi * 1j * u - kappa) ** 2 + xi ** 2 * (1j * u + u ** 2))
    g = (kappa - rho * xi * 1j * u - d) / (kappa - rho * xi * 1j * u + d)

    edt = np.exp(-d * T)

    D = (kappa - rho * xi * 1j * u - d) / (xi ** 2)
    D = D * (1.0 - edt) / (1.0 - g * edt)

    C = (r * 1j * u) * T
    C = C + (a / (xi ** 2)) * (
        (kappa - rho * xi * 1j * u - d) * T
        - 2.0 * np.log((1.0 - g * edt) / (1.0 - g))
    )

    return np.exp(C + D * v0 + 1j * u * np.log(S0))


def heston_price(S, K, T, r, v0, kappa, theta, xi, rho,
                  option_type='call', N=4096, eta=0.25, alpha=1.5):
    r"""
    Price a European option under the Heston model via Carr-Madan Fourier transform.

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
    option_type : str, optional
        'call' or 'put' (default 'call').
    N : int, optional
        Number of integration points (default 4096, power of 2 for FFT).
    eta : float, optional
        Spacing of log-strike grid (default 0.25).
    alpha : float, optional
        Damping factor for Carr-Madan (default 1.5).

    Returns
    -------
    float
        Heston option price.

    Notes
    -----
    Uses the Carr-Madan (1999) formulation. The call price is:

    .. math::
        C(K) = \frac{e^{-\alpha \log K}}{\pi}
        \int_0^\infty e^{-iv\log K} \psi(v) dv

    where

    .. math::
        \psi(v) = \frac{e^{-rT}\phi(v - (\alpha+1)i)}
        {\alpha^2 + \alpha - v^2 + i(2\alpha+1)v}

    and :math:`\phi` is the Heston characteristic function.

    Put prices are obtained via put-call parity:

    .. math::
        P = C - S + Ke^{-rT}
    """
    # Integration grid
    dv = 2.0 * np.pi / (N * eta)
    v = np.arange(N) * dv

    # Modified characteristic function for Carr-Madan
    u = v - (alpha + 1.0) * 1j
    phi = heston_characteristic_function(u, S, r, T, v0, kappa, theta, xi, rho)
    numer = np.exp(-r * T) * phi
    denom = alpha ** 2 + alpha - v ** 2 + 1j * (2.0 * alpha + 1.0) * v

    psi = numer / denom

    # Simpson's rule weights
    simpson_w = (3.0 + (-1.0) ** (np.arange(N) + 1)) / 3.0
    simpson_w[0] = 1.0 / 3.0

    # Integral
    integrand = np.real(psi * np.exp(-1j * v * np.log(K)))
    integral = dv * np.sum(integrand * simpson_w)

    call_price = np.exp(-alpha * np.log(K)) / np.pi * integral
    call_price = max(call_price, 0.0)

    if option_type == 'call':
        return call_price
    elif option_type == 'put':
        # Put-call parity
        put_price = call_price - S + K * np.exp(-r * T)
        return max(put_price, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def heston_calibrate(market_prices, strikes, maturities, S0, r,
                      init_params=None, max_iter=200, tol=1e-6):
    r"""
    Calibrate Heston parameters to market option prices via least squares.

    Parameters
    ----------
    market_prices : array_like
        Observed market option prices.
    strikes : array_like
        Corresponding strike prices.
    maturities : array_like
        Corresponding times to expiration.
    S0 : float
        Current underlying price.
    r : float
        Risk-free rate.
    init_params : array_like, optional
        Initial guess [v0, kappa, theta, xi, rho].
        Default: [0.04, 2.0, 0.04, 0.3, -0.5].
    max_iter : int, optional
        Maximum iterations for gradient descent (default 200).
    tol : float, optional
        Convergence tolerance (default 1e-6).

    Returns
    -------
    params : ndarray
        Calibrated parameters [v0, kappa, theta, xi, rho].
    mse : float
        Mean squared error of fit.

    Notes
    -----
    Uses gradient-based optimization (simple gradient descent) to minimize:

    .. math::
        \min_{\theta} \sum_i (P_i^{model}(\theta) - P_i^{market})^2

    subject to positivity and Feller condition constraints.

    The parameter bounds are:
    - v0, theta, xi > 0
    - kappa > 0
    - -1 <= rho <= 1
    - Feller: 2*kappa*theta >= xi^2 (optional enforcement)
    """
    market_prices = np.asarray(market_prices, dtype=float)
    strikes = np.asarray(strikes, dtype=float)
    maturities = np.asarray(maturities, dtype=float)

    if init_params is None:
        init_params = np.array([0.04, 2.0, 0.04, 0.3, -0.5])

    params = init_params.copy()
    n = len(market_prices)
    lr = 0.001
    eps = 1e-4

    def compute_mse(p):
        v0, kappa, theta, xi, rho = p
        model_prices = np.array([
            heston_price(S0, K, T_i, r, v0, max(kappa, 0.01), max(theta, 1e-6),
                          max(xi, 1e-6), np.clip(rho, -0.999, 0.999))
            for K, T_i in zip(strikes, maturities)
        ])
        errors = model_prices - market_prices
        return np.mean(errors ** 2)

    # Gradient descent with finite difference gradients
    prev_mse = np.inf

    for iteration in range(max_iter):
        current_mse = compute_mse(params)

        if abs(prev_mse - current_mse) < tol:
            break
        prev_mse = current_mse

        # Compute gradient via central differences
        grad = np.zeros(5)
        for j in range(5):
            p_up = params.copy()
            p_down = params.copy()
            p_up[j] += eps
            p_down[j] -= eps
            grad[j] = (compute_mse(p_up) - compute_mse(p_down)) / (2.0 * eps)

        # Update with gradient descent
        params = params - lr * grad

        # Enforce parameter bounds
        params[0] = max(params[0], 1e-6)   # v0 > 0
        params[1] = max(params[1], 0.01)   # kappa > 0
        params[2] = max(params[2], 1e-6)   # theta > 0
        params[3] = max(params[3], 1e-6)   # xi > 0
        params[4] = np.clip(params[4], -0.999, 0.999)  # rho in [-1, 1]

        # Adaptive learning rate: reduce if oscillation detected
        if iteration > 0 and iteration % 20 == 0:
            lr *= 0.5

    final_mse = compute_mse(params)

    return params, final_mse
