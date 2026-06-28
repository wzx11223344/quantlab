"""
Black-Litterman model for portfolio optimization.

Implements the Black-Litterman (1992) framework for combining market
equilibrium returns with subjective investor views to produce posterior
expected returns and covariance.

References
----------
Black, F. and Litterman, R. (1992). "Global Portfolio Optimization."
Financial Analysts Journal, 48(5), 28-43.
He, G. and Litterman, R. (1999). "The Intuition Behind Black-Litterman
Model Portfolios." Goldman Sachs Investment Management Research.
"""

import numpy as np


def black_litterman(prior_returns, cov_matrix, P, Q, tau=0.025, Omega=None):
    r"""
    Compute posterior expected returns and covariance using the
    Black-Litterman model.

    Parameters
    ----------
    prior_returns : ndarray of shape (n_assets,)
        Prior expected returns (typically from market equilibrium,
        e.g., implied from market-cap weights).
    cov_matrix : ndarray of shape (n_assets, n_assets)
        Covariance matrix of asset returns.
    P : ndarray of shape (n_views, n_assets)
        Pick matrix specifying which assets are involved in each view.
        Each row is a view: P[i,j] = +1 for long, -1 for short,
        0 for assets not in the view.
    Q : ndarray of shape (n_views,)
        View vector: expected returns for each view (e.g., 0.05 means
        5% expected return for that view portfolio).
    tau : float, optional
        Uncertainty scaling factor for the prior covariance.
        Small tau means high confidence in prior; default 0.025.
    Omega : ndarray of shape (n_views, n_views), optional
        Covariance matrix of view errors (confidence in views).
        If None (default), set to :math:`\tau \cdot P \Sigma P^T` (proportional).

    Returns
    -------
    posterior_returns : ndarray of shape (n_assets,)
        Posterior expected returns.
    posterior_cov : ndarray of shape (n_assets, n_assets)
        Posterior covariance matrix.

    Notes
    -----
    The Black-Litterman model combines two sources of information:

    1. **Prior (market equilibrium)** returns:
       :math:`\mu \sim \mathcal{N}(\Pi, \tau\Sigma)`

    2. **Investor views**:
       :math:`P\mu = Q + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \Omega)`

    The posterior distribution is:

    .. math::
        \mu_{BL} &= \left[(\tau\Sigma)^{-1} + P^T\Omega^{-1}P\right]^{-1}
        \left[(\tau\Sigma)^{-1}\Pi + P^T\Omega^{-1}Q\right] \\
        \Sigma_{BL} &= \Sigma + \left[(\tau\Sigma)^{-1} +
        P^T\Omega^{-1}P\right]^{-1}

    When Omega is not provided, it defaults to:

    .. math::
        \Omega = \text{diag}(\tau \cdot P \Sigma P^T)

    which assumes view uncertainty is proportional to prior uncertainty.

    Examples
    --------
    >>> prior = np.array([0.08, 0.06, 0.07])  # 3 assets
    >>> cov = np.eye(3) * 0.04
    >>> P = np.array([[1, -1, 0]])  # View: Asset 1 will outperform Asset 2
    >>> Q = np.array([0.03])         # By 3%
    >>> post_ret, post_cov = black_litterman(prior, cov, P, Q)
    """
    prior_returns = np.asarray(prior_returns, dtype=float)
    cov_matrix = np.asarray(cov_matrix, dtype=float)
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)

    n_assets = len(prior_returns)

    # Verify shapes
    if P.shape[1] != n_assets:
        raise ValueError(f"P must have {n_assets} columns, got {P.shape[1]}")
    if len(Q) != P.shape[0]:
        raise ValueError(f"Q length ({len(Q)}) must match P rows ({P.shape[0]})")
    if cov_matrix.shape != (n_assets, n_assets):
        raise ValueError(f"cov_matrix must be ({n_assets}, {n_assets})")

    # Prior covariance scaled by tau
    tau_Sigma = tau * cov_matrix

    # If Omega not provided, use proportional specification
    if Omega is None:
        # Each view's variance is proportional to the prior variance of that view
        view_variances = tau * np.diag(P @ cov_matrix @ P.T)
        # Add small floor to avoid numerical issues
        view_variances = np.maximum(view_variances, 1e-10)
        Omega_inv = np.diag(1.0 / view_variances)
    else:
        Omega = np.asarray(Omega, dtype=float)
        if Omega.shape != (len(Q), len(Q)):
            raise ValueError(f"Omega must be ({len(Q)}, {len(Q)})")
        try:
            Omega_inv = np.linalg.inv(Omega)
        except np.linalg.LinAlgError:
            Omega_inv = np.linalg.pinv(Omega)

    # Posterior precision
    try:
        prior_precision = np.linalg.inv(tau_Sigma)
    except np.linalg.LinAlgError:
        prior_precision = np.linalg.pinv(tau_Sigma)

    posterior_precision = prior_precision + P.T @ Omega_inv @ P

    # Posterior mean
    try:
        posterior_precision_inv = np.linalg.inv(posterior_precision)
    except np.linalg.LinAlgError:
        posterior_precision_inv = np.linalg.pinv(posterior_precision)

    posterior_returns = posterior_precision_inv @ (
        prior_precision @ prior_returns + P.T @ Omega_inv @ Q
    )

    # Posterior covariance
    posterior_cov = cov_matrix + posterior_precision_inv

    return posterior_returns, posterior_cov


def implied_equilibrium_returns(cov_matrix, market_weights, risk_aversion=2.5):
    r"""
    Compute implied equilibrium returns from market weights (reverse optimization).

    Parameters
    ----------
    cov_matrix : ndarray
        Covariance matrix of asset returns.
    market_weights : ndarray
        Market capitalization weights.
    risk_aversion : float, optional
        Risk aversion coefficient (default 2.5).

    Returns
    -------
    ndarray
        Implied equilibrium excess returns: :math:`\Pi = \lambda \Sigma w_{mkt}`.

    Notes
    -----
    Under the CAPM equilibrium, the market portfolio is optimal. Given
    market weights :math:`w_{mkt}`:

    .. math::
        \Pi = \lambda \Sigma w_{mkt}

    where :math:`\lambda` is the risk aversion parameter, typically 2-4.
    """
    cov = np.asarray(cov_matrix, dtype=float)
    market_weights = np.asarray(market_weights, dtype=float)

    return risk_aversion * (cov @ market_weights)
