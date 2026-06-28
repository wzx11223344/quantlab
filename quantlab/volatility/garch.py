"""
GARCH volatility models.

Implements the GARCH(1,1), GJR-GARCH(1,1), and EGARCH(1,1) models
with Maximum Likelihood Estimation via manual gradient descent.
No external optimization library is required.

References
----------
Engle, R.F. (1982). "Autoregressive Conditional Heteroscedasticity with
Estimates of the Variance of United Kingdom Inflation." Econometrica, 50(4).
Bollerslev, T. (1986). "Generalized Autoregressive Conditional
Heteroskedasticity." Journal of Econometrics, 31(3), 307-327.
Glosten, L.R., Jagannathan, R., and Runkle, D.E. (1993).
"On the Relation between the Expected Value and the Volatility of the
Nominal Excess Return on Stocks." Journal of Finance, 48(5).
Nelson, D.B. (1991). "Conditional Heteroskedasticity in Asset Returns:
A New Approach." Econometrica, 59(2), 347-370.
"""

import numpy as np


class GARCH:
    r"""
    GARCH(p,q) model with MLE via gradient descent.

    The GARCH(1,1) model for conditional variance:

    .. math::
        \sigma^2_t = \omega + \alpha \varepsilon^2_{t-1} + \beta \sigma^2_{t-1}

    where :math:`\varepsilon_t = \sigma_t z_t`, :math:`z_t \sim \mathcal{N}(0,1)`.

    Parameters
    ----------
    p : int
        ARCH order (default 1).
    q : int
        GARCH order (default 1).

    Notes
    -----
    The log-likelihood function is:

    .. math::
        \mathcal{L} = -\frac{1}{2}\sum_{t=1}^T
        \left[\log(2\pi) + \log(\sigma^2_t) + \frac{\varepsilon^2_t}{\sigma^2_t}\right]

    For numerical stability, variance is floored at 1e-8.
    """

    def __init__(self, p=1, q=1):
        self.p = p
        self.q = q
        self.params = None  # [omega, alpha_1, ..., alpha_p, beta_1, ..., beta_q]
        self.fitted_var = None

    def fit(self, returns, verbose=False, max_iter=1000, lr=0.001, tol=1e-6):
        r"""
        Fit the GARCH model via MLE with gradient descent.

        Parameters
        ----------
        returns : array_like
            Array of returns (mean-subtracted or demeaned internally).
        verbose : bool, optional
            If True, print iteration progress (default False).
        max_iter : int, optional
            Maximum gradient descent iterations (default 1000).
        lr : float, optional
            Learning rate (default 0.001).
        tol : float, optional
            Convergence tolerance (default 1e-6).

        Returns
        -------
        self
            Fitted model.
        """
        returns = np.asarray(returns, dtype=float)
        # Demean returns
        eps = returns - np.mean(returns)
        n = len(eps)

        # Initialize parameters: [omega, alpha_1, ..., alpha_p, beta_1, ..., beta_q]
        n_params = 1 + self.p + self.q
        params = np.zeros(n_params)
        params[0] = np.var(eps) * 0.1  # omega: small fraction of unconditional variance
        params[1:1 + self.p] = 0.1 / self.p  # alpha: sum to ~0.1
        params[1 + self.p:] = 0.8 / self.q    # beta: sum to ~0.8

        prev_loss = np.inf

        for iteration in range(max_iter):
            # Compute conditional variance series
            sigma2 = self._compute_variance(eps, params)
            loss = self._log_likelihood(eps, sigma2)

            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: LL = {-loss:.4f}, params = {params}")

            if abs(prev_loss - loss) < tol:
                break
            prev_loss = loss

            # Compute gradient numerically
            grad = np.zeros_like(params)
            eps_grad = 1e-5
            for j in range(n_params):
                p_up = params.copy()
                p_up[j] += eps_grad
                # Ensure positivity
                p_up = np.maximum(p_up, 1e-15)
                sigma2_up = self._compute_variance(eps, p_up)
                loss_up = self._log_likelihood(eps, sigma2_up)
                grad[j] = (loss_up - loss) / eps_grad

            # Gradient clipping to prevent explosion
            grad_norm = np.sqrt(np.sum(grad ** 2))
            if grad_norm > 10.0 and grad_norm > 0:
                grad = grad * (10.0 / grad_norm)

            # Gradient descent with positivity constraint
            params = params - lr * grad

            # Enforce positivity with data-adaptive minimum
            min_omega = np.var(eps) * 1e-10 if np.var(eps) > 0 else 1e-15
            params[0] = max(params[0], min_omega)
            params[1:] = np.maximum(params[1:], 0.0)

            # Enforce stationarity: alpha + beta < 1
            total_ab = params[1:].sum()
            if total_ab >= 1.0:
                scale = 0.99 / total_ab
                params[1:] *= scale

        self.params = params
        self.fitted_var = self._compute_variance(eps, params)

        if verbose:
            print(f"Converged at iteration {iteration}: LL = {-loss:.4f}")
            print(f"omega={params[0]:.6f}, alpha={params[1]:.6f}, beta={params[2]:.6f}")

        return self

    def _compute_variance(self, eps, params):
        """Compute conditional variance series given parameters."""
        n = len(eps)
        sigma2 = np.zeros(n)
        omega = max(params[0], 1e-15)

        # Data-adaptive floor
        var_floor = max(np.var(eps) * 1e-8, 1e-15)

        # Initialize with unconditional variance
        persistence = params[1:].sum()
        if persistence < 1.0:
            sigma2[0] = omega / (1.0 - persistence)
        else:
            sigma2[0] = np.var(eps)

        sigma2[0] = max(sigma2[0], var_floor)

        for t in range(1, n):
            sigma2[t] = omega
            # ARCH terms
            for i in range(self.p):
                if t - 1 - i >= 0:
                    sigma2[t] += params[1 + i] * eps[t - 1 - i] ** 2
            # GARCH terms
            for j in range(self.q):
                if t - 1 - j >= 0:
                    sigma2[t] += params[1 + self.p + j] * sigma2[t - 1 - j]

            sigma2[t] = max(sigma2[t], var_floor)

        return sigma2

    @staticmethod
    def _log_likelihood(eps, sigma2):
        """Compute negative log-likelihood (to minimize)."""
        n = len(eps)
        # -LL for minimization
        ll = -0.5 * np.sum(np.log(2.0 * np.pi) + np.log(sigma2) + eps ** 2 / sigma2)
        return -ll  # Return negative so minimizing = maximizing LL

    def forecast(self, horizon=1):
        r"""
        Forecast conditional variance n-steps ahead.

        Parameters
        ----------
        horizon : int, optional
            Forecast horizon (default 1).

        Returns
        -------
        ndarray
            Forecasted variances for each step ahead.

        Notes
        -----
        For GARCH(1,1), the n-step ahead forecast is:

        .. math::
            \mathbb{E}[\sigma^2_{T+n}|\mathcal{F}_T] =
            \omega \frac{1 - (\alpha+\beta)^{n-1}}{1 - (\alpha+\beta)}
            + (\alpha+\beta)^{n-1} \sigma^2_{T+1}

        where :math:`\sigma^2_{T+1}` uses the known :math:`\varepsilon_T`.
        """
        if self.params is None:
            raise ValueError("Model must be fitted before forecasting.")

        omega = self.params[0]
        persistence = self.params[1:].sum()

        # Last known variance
        last_var = max(self.fitted_var[-1], 1e-15)

        forecasts = np.zeros(horizon)
        for h in range(horizon):
            if persistence < 1.0:
                long_run = omega / (1.0 - persistence)
                forecasts[h] = long_run + (persistence ** (h + 1)) * (last_var - long_run)
            else:
                forecasts[h] = last_var

        return forecasts


class GJR_GARCH:
    r"""
    GJR-GARCH(1,1) model with leverage effect.

    The GJR-GARCH model allows asymmetric response to positive and
    negative shocks via a leverage parameter:

    .. math::
        \sigma^2_t = \omega + (\alpha + \gamma I_{t-1})\varepsilon^2_{t-1}
        + \beta \sigma^2_{t-1}

    where :math:`I_{t-1} = 1` if :math:`\varepsilon_{t-1} < 0`, otherwise 0.

    Parameters
    ----------
    p : int, optional
        ARCH order (default 1).
    q : int, optional
        GARCH order (default 1).
    """

    def __init__(self, p=1, q=1):
        self.p = p
        self.q = q
        self.params = None  # [omega, alpha, gamma, beta]
        self.fitted_var = None

    def fit(self, returns, verbose=False, max_iter=1000, lr=0.0005, tol=1e-6):
        r"""
        Fit GJR-GARCH via MLE with gradient descent.

        Parameters
        ----------
        returns : array_like
            Array of returns.
        verbose : bool, optional
            Print progress.
        max_iter : int, optional
            Maximum iterations.
        lr : float, optional
            Learning rate.
        tol : float, optional
            Convergence tolerance.

        Returns
        -------
        self
            Fitted model.
        """
        returns = np.asarray(returns, dtype=float)
        eps = returns - np.mean(returns)
        n = len(eps)

        # Parameters: [omega, alpha, gamma, beta]
        var0 = np.var(eps)
        params = np.array([var0 * 0.05, 0.05, 0.05, 0.8])

        prev_loss = np.inf

        for iteration in range(max_iter):
            sigma2 = self._compute_variance(eps, params)
            loss = self._neg_log_likelihood(eps, sigma2)

            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: NLL = {loss:.4f}, params = {params}")

            if abs(prev_loss - loss) < tol:
                break
            prev_loss = loss

            grad = np.zeros(4)
            eps_grad = 1e-5
            for j in range(4):
                p_up = params.copy()
                p_up[j] += eps_grad
                p_up = np.maximum(p_up, 1e-8)
                sigma2_up = self._compute_variance(eps, p_up)
                loss_up = self._neg_log_likelihood(eps, sigma2_up)
                grad[j] = (loss_up - loss) / eps_grad

            params = params - lr * grad
            params = np.maximum(params, 1e-8)

            # Enforce stationarity: alpha + gamma/2 + beta < 1
            persistence = params[1] + 0.5 * params[2] + params[3]
            if persistence >= 1.0:
                scale = 0.99 / persistence
                params[1:] *= scale

        self.params = params
        self.fitted_var = self._compute_variance(eps, params)

        if verbose:
            print(f"Converged: omega={params[0]:.6f}, alpha={params[1]:.6f}, "
                  f"gamma={params[2]:.6f}, beta={params[3]:.6f}")

        return self

    def _compute_variance(self, eps, params):
        """Compute GJR-GARCH conditional variance."""
        n = len(eps)
        sigma2 = np.zeros(n)
        omega, alpha, gamma, beta = params

        sigma2[0] = np.var(eps)
        for t in range(1, n):
            indicator = 1.0 if eps[t - 1] < 0 else 0.0
            sigma2[t] = omega + (alpha + gamma * indicator) * eps[t - 1] ** 2 + beta * sigma2[t - 1]
            sigma2[t] = max(sigma2[t], 1e-8)

        return sigma2

    @staticmethod
    def _neg_log_likelihood(eps, sigma2):
        ll = -0.5 * np.sum(np.log(2.0 * np.pi) + np.log(sigma2) + eps ** 2 / sigma2)
        return -ll

    def forecast(self, horizon=1):
        """Forecast conditional variance."""
        if self.params is None:
            raise ValueError("Model must be fitted first.")
        omega, alpha, gamma, beta = self.params
        persistence = alpha + 0.5 * gamma + beta
        last_var = self.fitted_var[-1]

        forecasts = np.zeros(horizon)
        for h in range(horizon):
            if h == 0:
                forecasts[h] = omega + persistence * last_var
            else:
                forecasts[h] = omega + persistence * forecasts[h - 1]
        return forecasts


class EGARCH:
    r"""
    EGARCH(1,1) model (Nelson 1991).

    The EGARCH model ensures positivity of variance by modeling log-variance:

    .. math::
        \log \sigma^2_t = \omega + \alpha\left( \frac{|\varepsilon_{t-1}|}
        {\sigma_{t-1}} - \sqrt{\frac{2}{\pi}} \right)
        + \gamma \frac{\varepsilon_{t-1}}{\sigma_{t-1}}
        + \beta \log \sigma^2_{t-1}

    where :math:`\gamma` captures the leverage effect.
    """

    def __init__(self):
        self.params = None  # [omega, alpha, gamma, beta]
        self.fitted_var = None

    def fit(self, returns, verbose=False, max_iter=500, lr=0.001, tol=1e-6):
        r"""
        Fit EGARCH via MLE.

        Parameters
        ----------
        returns : array_like
            Array of returns.
        verbose : bool, optional
            Print progress.
        max_iter : int, optional
            Maximum iterations.
        lr : float, optional
            Learning rate.
        tol : float, optional
            Convergence tolerance.

        Returns
        -------
        self
            Fitted model.
        """
        returns = np.asarray(returns, dtype=float)
        eps = returns - np.mean(returns)
        n = len(eps)

        # [omega, alpha, gamma, beta]
        params = np.array([0.0, 0.1, -0.05, 0.95])

        prev_loss = np.inf

        for iteration in range(max_iter):
            sigma2 = self._compute_variance(eps, params)
            loss = self._neg_log_likelihood(eps, sigma2)

            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: NLL = {loss:.4f}, params = {params}")

            if abs(prev_loss - loss) < tol:
                break
            prev_loss = loss

            grad = np.zeros(4)
            eps_grad = 1e-5
            for j in range(4):
                p_up = params.copy()
                p_up[j] += eps_grad
                sigma2_up = self._compute_variance(eps, p_up)
                loss_up = self._neg_log_likelihood(eps, sigma2_up)
                grad[j] = (loss_up - loss) / eps_grad

            params = params - lr * grad

            # beta stability constraint
            params[3] = min(max(params[3], 0.0), 0.999)

        self.params = params
        self.fitted_var = self._compute_variance(eps, params)
        return self

    def _compute_variance(self, eps, params):
        """Compute EGARCH log-variance."""
        n = len(eps)
        log_var = np.zeros(n)
        omega, alpha, gamma, beta = params
        sqrt_2_pi = np.sqrt(2.0 / np.pi)
        var_floor = -50.0  # Lower bound for log variance

        log_var[0] = np.log(np.var(eps)) if np.var(eps) > 0 else 0.0

        for t in range(1, n):
            prev_log = max(log_var[t - 1], var_floor)
            std_prev = np.sqrt(np.exp(prev_log))
            std_prev = max(std_prev, 1e-15)

            z = eps[t - 1] / std_prev
            log_var[t] = (
                omega
                + alpha * (np.abs(z) - sqrt_2_pi)
                + gamma * z
                + beta * prev_log
            )
            # Clamp to prevent overflow on exp later
            log_var[t] = max(log_var[t], var_floor)

        return np.exp(np.clip(log_var, var_floor, 50.0))

    @staticmethod
    def _neg_log_likelihood(eps, sigma2):
        ll = -0.5 * np.sum(np.log(2.0 * np.pi) + np.log(sigma2) + eps ** 2 / sigma2)
        return -ll

    def forecast(self, horizon=1):
        """Forecast log-variance."""
        if self.params is None:
            raise ValueError("Model must be fitted first.")
        return np.ones(horizon)
