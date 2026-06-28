"""
Risk Parity and Hierarchical Risk Parity (HRP) portfolio construction.

Implements equal risk contribution (ERC) portfolios using the cyclical
coordinate descent algorithm (Spinu 2013) and a simplified Hierarchical
Risk Parity approach (Lopez de Prado 2016).

References
----------
Spinu, F. (2013). "An Algorithm for Computing Risk Parity Weights."
Available at SSRN 2297383.
Lopez de Prado, M. (2016). "Building Diversified Portfolios that Outperform
Out of Sample." Journal of Portfolio Management, 42(4), 59-69.
"""

import numpy as np


def risk_parity_portfolio(cov_matrix, max_iter=1000, tol=1e-8):
    r"""
    Compute the Equal Risk Contribution (ERC) / Risk Parity portfolio
    using the cyclical coordinate descent algorithm (Spinu 2013).

    Parameters
    ----------
    cov_matrix : ndarray of shape (n_assets, n_assets)
        Covariance matrix of asset returns.
    max_iter : int, optional
        Maximum iterations (default 1000).
    tol : float, optional
        Convergence tolerance (default 1e-8).

    Returns
    -------
    ndarray
        Risk parity portfolio weights (sum = 1).

    Notes
    -----
    The ERC portfolio satisfies:

    .. math::
        w_i \frac{\partial \sigma(w)}{\partial w_i} =
        w_j \frac{\partial \sigma(w)}{\partial w_j}, \quad \forall i,j

    where :math:`\sigma(w) = \sqrt{w^T\Sigma w}` is portfolio volatility.

    The marginal risk contribution (MRC) of asset i is:

    .. math::
        MRC_i = \frac{(\Sigma w)_i}{\sqrt{w^T\Sigma w}}

    The risk contribution is :math:`RC_i = w_i \cdot MRC_i`.
    The ERC target is :math:`RC_i = \sigma(w) / n` for all i.

    The algorithm updates weights cyclically:

    .. math::
        w_i \leftarrow w_i + \lambda \cdot (\text{target} - RC_i) / MRC_i

    where :math:`\lambda` is a step size parameter.
    """
    cov = np.asarray(cov_matrix, dtype=float)
    n = len(cov)
    w = np.ones(n) / n  # Equal weight initial guess

    for iteration in range(max_iter):
        w_old = w.copy()

        # Portfolio volatility and marginal risks
        portfolio_var = w @ cov @ w
        portfolio_vol = np.sqrt(max(portfolio_var, 1e-15))
        MR = cov @ w / portfolio_vol  # Marginal risk contributions
        RC = w * MR                   # Risk contributions
        target_RC = portfolio_vol / n  # Equal risk contribution target

        # Update each weight via cyclical coordinate descent
        for i in range(n):
            # Solve for w_i that equalizes risk contribution
            # Using the quadratic formula for ERC
            sigma_i = np.sqrt(cov[i, i]) if cov[i, i] > 0 else 1e-8
            others = np.delete(np.arange(n), i)
            w_others = w[others]
            cov_i_others = cov[i, others]
            cov_others = cov[np.ix_(others, others)]

            # Compute terms for the update
            a = sigma_i ** 2
            b = cov_i_others @ w_others
            c_others_var = w_others @ cov_others @ w_others

            # Target risk contribution
            target_rc = target_RC

            # Solve for w_i: w_i^2 * sigma_i^2 + w_i * b = target_rc * portfolio_vol
            # After normalization, this gives the new weight
            if a > 0:
                # Rewriting: solve w_i * (w_i*a + b) / vol_new = target_rc
                # Approximate: use current vol
                if b ** 2 + 4 * a * target_rc * portfolio_vol >= 0:
                    discriminant = np.sqrt(b ** 2 + 4 * a * target_rc * portfolio_vol)
                    w_i_new = (-b + discriminant) / (2 * a)
                    w_i_new = max(w_i_new, 0.0)
                else:
                    w_i_new = w[i]
            else:
                w_i_new = w[i]

            w[i] = w_i_new

        # Normalize weights
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum
        else:
            w = np.ones(n) / n

        # Check convergence
        if np.max(np.abs(w - w_old)) < tol:
            break

    return w


def _pairwise_distance(cov):
    """Compute distance matrix from covariance."""
    n = len(cov)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rho = cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])
            dist[i, j] = np.sqrt(0.5 * (1 - rho))
    return dist


def _quasi_diagonalize(dist):
    """Recursively sort assets by distance proximity."""
    n = len(dist)
    if n <= 1:
        return list(range(n))

    # Find the two closest elements
    min_val = np.inf
    pair = (0, 1)
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] < min_val:
                min_val = dist[i, j]
                pair = (i, j)

    i, j = pair
    # Recursively cluster
    remaining = [k for k in range(n) if k != i and k != j]

    sorted_rem = _quasi_diagonalize(dist[np.ix_(remaining, remaining)])
    sorted_rem_mapped = [remaining[k] for k in sorted_rem]

    return [i] + [j] + sorted_rem_mapped


def _recursive_bisection(cov, sorted_indices):
    """Recursively bisect and compute weights."""
    n = len(sorted_indices)
    w = np.ones(n) / n

    if n <= 1:
        return w

    # Split into two clusters
    split = n // 2
    left_idx = sorted_indices[:split]
    right_idx = sorted_indices[split:]

    # Recursive allocation
    w_left = _recursive_bisection(
        cov[np.ix_(left_idx, left_idx)],
        list(range(len(left_idx)))
    )
    w_right = _recursive_bisection(
        cov[np.ix_(right_idx, right_idx)],
        list(range(len(right_idx)))
    )

    # Compute cluster variances
    var_left = w_left @ cov[np.ix_(left_idx, left_idx)] @ w_left
    var_right = w_right @ cov[np.ix_(right_idx, right_idx)] @ w_right

    # Inverse variance allocation
    alloc_left = (1.0 / var_left) / (1.0 / var_left + 1.0 / var_right) if var_left + var_right > 0 else 0.5
    alloc_right = 1.0 - alloc_left

    # Map back to original indices
    w_full = np.zeros(len(cov))
    for k, idx in enumerate(left_idx):
        w_full[idx] = alloc_left * w_left[k]
    for k, idx in enumerate(right_idx):
        w_full[idx] = alloc_right * w_right[k]

    return w_full


def hrp(cov_matrix):
    r"""
    Compute the Hierarchical Risk Parity (HRP) portfolio weights.

    This is a simplified implementation based on Lopez de Prado (2016).

    Parameters
    ----------
    cov_matrix : ndarray of shape (n_assets, n_assets)
        Covariance matrix.

    Returns
    -------
    ndarray
        HRP portfolio weights.

    Notes
    -----
    HRP addresses the instability of Markowitz optimization by:

    1. **Tree Clustering**: Build a hierarchical tree from the correlation matrix.
    2. **Quasi-Diagonalization**: Reorder assets to put similar ones together.
    3. **Recursive Bisection**: Allocate capital top-down using inverse-variance.

    This avoids matrix inversion entirely, making it robust for large
    portfolios and ill-conditioned covariance matrices.

    Steps:

    .. math::
        d_{ij} &= \sqrt{\frac{1}{2}(1 - \rho_{ij})} \\
        \text{Cluster} &\text{ using nearest-point algorithm} \\
        w_{left} &= \alpha w_{left}, \quad
        w_{right} = (1-\alpha) w_{right} \\
        \alpha &= \frac{1/\sigma^2_{left}}{1/\sigma^2_{left} + 1/\sigma^2_{right}}
    """
    cov = np.asarray(cov_matrix, dtype=float)
    n = len(cov)

    if n <= 1:
        return np.ones(n)

    # Step 1: Distance matrix
    dist = _pairwise_distance(cov)

    # Step 2: Quasi-diagonalization (reordering)
    order = _quasi_diagonalize(dist)

    # Step 3: Recursive bisection
    w = _recursive_bisection(cov, order)

    # Reorder weights to original order
    w_original = np.zeros(n)
    for i, idx in enumerate(order):
        w_original[idx] = w[i]

    return w_original
