# QuantLab — Quantitative Finance Laboratory

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**QuantLab** is a comprehensive, zero-dependency (beyond NumPy) open-source financial engineering toolkit. Every algorithm is implemented from scratch with mathematical rigor, making it ideal for learning, research, and production use.

## Features

### Options Pricing
| Method | Features |
|--------|----------|
| **Black-Scholes** | Exact formula, all Greeks (delta, gamma, theta, vega, rho), implied volatility via Newton-Raphson |
| **Binomial Tree** | CRR parameterization, European & American options, early exercise, Greeks via finite difference |
| **Monte Carlo** | Antithetic variates, control variates, Asian options, barrier options, Box-Muller sampling |
| **Heston Model** | Stochastic volatility, Carr-Madan Fourier transform pricing, least-squares calibration |

### Risk Management
- **Value at Risk (VaR)**: Historical, Parametric, Monte Carlo
- **Conditional VaR (CVaR)**: Expected shortfall
- **Backtesting**: Kupiec POF test, Christoffersen independence test
- **Risk Metrics**: Sharpe, Sortino, Calmar ratios, max drawdown, information ratio

### Portfolio Optimization
- **Markowitz Mean-Variance**: Efficient frontier, min variance, max Sharpe (KKT closed-form)
- **Black-Litterman**: Bayesian integration of investor views
- **Risk Parity**: Equal risk contribution (Spinu 2013), Hierarchical Risk Parity (HRP)

### Volatility Modeling
- **GARCH Family**: GARCH(1,1), GJR-GARCH, EGARCH with manual MLE
- **Realized Volatility**: Parkinson, Garman-Klass, Yang-Zhang estimators

### Yield Curve
- **Bootstrapping**: Spot curve from par yields
- **Models**: Nelson-Siegel (3-factor), Svensson (4-factor), least-squares calibration
- **Analytics**: Forward rates, Macaulay/modified duration, convexity

## Installation

```bash
pip install quantlab
```

Or from source:

```bash
git clone https://github.com/quantlab/quantlab.git
cd quantlab
pip install -e .
```

**Requirements**: Python 3.8+ and NumPy. That's it — no SciPy, no pandas, no cvxopt.

## Quick Start

### Price a European Call Option

```python
from quantlab.options import black_scholes_price, black_scholes_greeks

price = black_scholes_price(S=100, K=105, T=1.0, r=0.05, sigma=0.2, option_type='call')
print(f"Call price: {price:.4f}")

greeks = black_scholes_greeks(S=100, K=105, T=1.0, r=0.05, sigma=0.2)
print(f"Delta: {greeks['delta']:.4f}, Vega: {greeks['vega']:.4f}")
```

### Compute VaR and CVaR

```python
from quantlab.risk import historical_var, cvar
import numpy as np

returns = np.random.normal(0.001, 0.02, 1000)
var_95 = historical_var(returns, alpha=0.05)
cvar_95 = cvar(returns, alpha=0.05)
print(f"95% VaR: {var_95:.4f}, 95% CVaR: {cvar_95:.4f}")
```

### Build Efficient Frontier

```python
from quantlab.portfolio import efficient_frontier, max_sharpe_portfolio
import numpy as np

er = np.array([0.10, 0.12, 0.08])        # expected returns
cov = np.array([[0.04, 0.01, 0.005],
                [0.01, 0.09, 0.02],
                [0.005, 0.02, 0.03]])   # covariance matrix

ret, vol, w = efficient_frontier(er, cov, n_points=50)
w_msr = max_sharpe_portfolio(er, cov, rf=0.03)
print(f"Max Sharpe weights: {w_msr}")
```

### Fit GARCH and Forecast

```python
from quantlab.volatility import GARCH

garch = GARCH(p=1, q=1)
garch.fit(returns, verbose=True)
forecast = garch.forecast(horizon=10)
print(f"10-day vol forecast: {forecast}")
```

## Architecture

```
quantlab/
├── options/          Options pricing (BS, binomial, MC, Heston)
├── risk/             Risk management (VaR, CVaR, metrics)
├── portfolio/        Portfolio optimization (Markowitz, BL, risk parity)
├── volatility/       Volatility modeling (GARCH, realized)
├── yield_curve/      Yield curve construction and analytics
└── utils.py          Brownian motion, normality tests
```

## Mathematical Foundation

All implementations follow standard references:

- **Options**: Hull, J.C. *Options, Futures, and Other Derivatives*
- **Volatility**: Tsay, R.S. *Analysis of Financial Time Series*
- **Portfolio**: Meucci, A. *Risk and Asset Allocation*
- **Yield Curve**: James, J. and Webber, N. *Interest Rate Modelling*

## License

MIT License — see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request. Areas of interest:
- More exotic option types (lookback, chooser, compound)
- SABR model
- Copula-based risk aggregation
- Extended GARCH variants (FIGARCH, DCC-GARCH)
- GPU-accelerated Monte Carlo
