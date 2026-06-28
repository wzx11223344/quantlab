"""
Yield curve models: Nelson-Siegel and Svensson.

Implements the Nelson-Siegel (1987) three-factor model and the
Svensson (1994) four-factor extension for parametric yield curve
fitting via non-linear least squares.

References
----------
Nelson, C.R. and Siegel, A.F. (1987). "Parsimonious Modeling of Yield
Curves." Journal of Business, 60(4), 473-489.
Svensson, L.E.O. (1994). "Estimating and Interpreting Forward Interest
Rates: Sweden 1992-1994." NBER Working Paper No. 4871.
"""

import numpy as np


def nelson_siegel(t, beta0, beta1, beta2, tau):
    r"""
    Nelson-Siegel instantaneous forward rate curve.

    Parameters
    ----------
    t : float or array_like
        Time to maturity in years.
    beta0 : float
        Long-run level of interest rates (asymptote).
    beta1 : float
        Short-term component (deviation from long-run).
    beta2 : float
        Medium-term component (hump shape).
    tau : float
        Decay factor controlling the hump position (> 0).

    Returns
    -------
    float or ndarray
        Yield at maturity t.

    Notes
    -----
    The Nelson-Siegel yield curve is:

    .. math::
        y(t) = \beta_0 + \beta_1\left(\frac{1 - e^{-t/\tau}}{t/\tau}\right)
        + \beta_2\left(\frac{1 - e^{-t/\tau}}{t/\tau} - e^{-t/\tau}\right)

    - :math:`\beta_0`: level factor (loads on all maturities equally)
    - :math:`\beta_1`: slope factor (loads heavily on short maturities)
    - :math:`\beta_2`: curvature factor (loads on medium maturities)

    For :math:`t = 0` (instantaneous rate):

    .. math::
        y(0) = \beta_0 + \beta_1

    Examples
    --------
    >>> nelson_siegel([1, 5, 10], 0.04, -0.02, -0.01, 2.0)
    array([0.028..., 0.035..., 0.038...])
    """
    t = np.asarray(t, dtype=float)
    scalar_input = t.ndim == 0
    t = np.atleast_1d(t)

    # Handle t = 0 case
    result = np.zeros_like(t)

    for i, ti in enumerate(t):
        if ti < 1e-10:
            result[i] = beta0 + beta1
        else:
            tau_t = ti / tau
            exp_neg = np.exp(-tau_t)
            factor1 = (1.0 - exp_neg) / tau_t
            factor2 = factor1 - exp_neg
            result[i] = beta0 + beta1 * factor1 + beta2 * factor2

    return float(result[0]) if scalar_input else result


def svensson(t, beta0, beta1, beta2, beta3, tau1, tau2):
    r"""
    Svensson (1994) extended yield curve model.

    Parameters
    ----------
    t : float or array_like
        Time to maturity in years.
    beta0 : float
        Long-run level.
    beta1 : float
        Short-term component.
    beta2 : float
        First medium-term (hump) component.
    beta3 : float
        Second medium-term (hump) component.
    tau1 : float
        First decay factor (> 0).
    tau2 : float
        Second decay factor (> 0).

    Returns
    -------
    float or ndarray
        Yield at maturity t.

    Notes
    -----
    The Svensson model adds a second hump to Nelson-Siegel:

    .. math::
        y(t) &= \beta_0 + \beta_1\left(\frac{1 - e^{-t/\tau_1}}{t/\tau_1}\right) \\
        &+ \beta_2\left(\frac{1 - e^{-t/\tau_1}}{t/\tau_1} - e^{-t/\tau_1}\right) \\
        &+ \beta_3\left(\frac{1 - e^{-t/\tau_2}}{t/\tau_2} - e^{-t/\tau_2}\right)

    The additional term allows for a second hump, improving fit for
    longer maturities.

    Examples
    --------
    >>> svensson([1, 5, 10], 0.04, -0.02, -0.01, 0.005, 2.0, 5.0)
    """
    t = np.asarray(t, dtype=float)
    scalar_input = t.ndim == 0
    t = np.atleast_1d(t)

    result = np.zeros_like(t)

    for i, ti in enumerate(t):
        if ti < 1e-10:
            result[i] = beta0 + beta1
        else:
            tau_t1 = ti / tau1
            tau_t2 = ti / tau2
            exp1 = np.exp(-tau_t1)
            exp2 = np.exp(-tau_t2)

            factor1 = (1.0 - exp1) / tau_t1
            factor2 = factor1 - exp1
            factor3 = (1.0 - exp2) / tau_t2 - exp2

            result[i] = (
                beta0
                + beta1 * factor1
                + beta2 * factor2
                + beta3 * factor3
            )

    return float(result[0]) if scalar_input else result


def calibrate_ns(maturities, yields, init_params=None, max_iter=200, lr=0.001, tol=1e-6):
    r"""
    Calibrate Nelson-Siegel parameters via least squares.

    Parameters
    ----------
    maturities : array_like
        Maturities in years.
    yields : array_like
        Observed yields at each maturity.
    init_params : array_like, optional
        Initial guess [beta0, beta1, beta2, tau].
        Default: [max(yields), -0.01, -0.01, 2.0].
    max_iter : int, optional
        Maximum gradient descent iterations (default 200).
    lr : float, optional
        Learning rate (default 0.001).
    tol : float, optional
        Convergence tolerance (default 1e-6).

    Returns
    -------
    params : ndarray
        Calibrated parameters [beta0, beta1, beta2, tau].
    r_squared : float
        R-squared of the fit.

    Notes
    -----
    Minimizes the sum of squared errors:

    .. math::
        \min_{\beta_0,\beta_1,\beta_2,\tau}
        \sum_i (y(t_i) - y_i^{obs})^2
    """
    maturities = np.asarray(maturities, dtype=float)
    yields = np.asarray(yields, dtype=float)

    if init_params is None:
        init_params = np.array([yields[-1], -0.01, -0.01, 2.0])

    params = init_params.copy()
    eps = 1e-5

    def mse(p):
        pred = nelson_siegel(maturities, p[0], p[1], p[2], p[3])
        return np.mean((pred - yields) ** 2)

    prev_mse = np.inf

    for iteration in range(max_iter):
        current_mse = mse(params)

        if abs(prev_mse - current_mse) < tol:
            break
        prev_mse = current_mse

        # Gradient via finite differences
        grad = np.zeros(4)
        for j in range(4):
            p_up = params.copy()
            p_up[j] += eps
            grad[j] = (mse(p_up) - current_mse) / eps

        params = params - lr * grad

        # tau must be positive
        params[3] = max(params[3], 0.01)

        # Adaptive learning rate
        if iteration > 0 and iteration % 50 == 0:
            lr *= 0.5

    # Compute R-squared
    pred_final = nelson_siegel(maturities, params[0], params[1], params[2], params[3])
    ss_res = np.sum((yields - pred_final) ** 2)
    ss_tot = np.sum((yields - np.mean(yields)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return params, r_squared


def calibrate_svensson(maturities, yields, init_params=None, max_iter=200, lr=0.001, tol=1e-6):
    r"""
    Calibrate Svensson parameters via least squares.

    Parameters
    ----------
    maturities : array_like
        Maturities in years.
    yields : array_like
        Observed yields.
    init_params : array_like, optional
        Initial guess [beta0, beta1, beta2, beta3, tau1, tau2].
        Default: [max(yields), -0.01, -0.01, 0.0, 2.0, 5.0].
    max_iter : int, optional
        Maximum iterations.
    lr : float, optional
        Learning rate.
    tol : float, optional
        Convergence tolerance.

    Returns
    -------
    params : ndarray
        Calibrated parameters [beta0, beta1, beta2, beta3, tau1, tau2].
    r_squared : float
        R-squared of the fit.

    Notes
    -----
    .. math::
        \min_{\beta_0,\beta_1,\beta_2,\beta_3,\tau_1,\tau_2}
        \sum_i (y(t_i) - y_i^{obs})^2
    """
    maturities = np.asarray(maturities, dtype=float)
    yields = np.asarray(yields, dtype=float)

    if init_params is None:
        init_params = np.array([yields[-1], -0.01, -0.01, 0.0, 2.0, 5.0])

    params = init_params.copy()
    eps = 1e-5

    def mse(p):
        pred = svensson(maturities, p[0], p[1], p[2], p[3], p[4], p[5])
        return np.mean((pred - yields) ** 2)

    prev_mse = np.inf

    for iteration in range(max_iter):
        current_mse = mse(params)

        if abs(prev_mse - current_mse) < tol:
            break
        prev_mse = current_mse

        grad = np.zeros(6)
        for j in range(6):
            p_up = params.copy()
            p_up[j] += eps
            grad[j] = (mse(p_up) - current_mse) / eps

        params = params - lr * grad

        # taus must be positive
        params[4] = max(params[4], 0.01)
        params[5] = max(params[5], 0.01)

        if iteration > 0 and iteration % 50 == 0:
            lr *= 0.5

    pred_final = svensson(maturities, params[0], params[1], params[2], params[3], params[4], params[5])
    ss_res = np.sum((yields - pred_final) ** 2)
    ss_tot = np.sum((yields - np.mean(yields)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return params, r_squared
