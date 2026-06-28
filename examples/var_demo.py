"""
VaR and Risk Metrics Demo

Demonstrates Value at Risk (VaR) computation using three methodologies,
Conditional VaR (CVaR), VaR backtesting, and risk-adjusted performance metrics.

Run: python examples/var_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from quantlab.risk import (
    historical_var,
    parametric_var,
    monte_carlo_var,
    cvar,
    backtest_var,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    information_ratio,
)


def main():
    print("=" * 70)
    print("QuantLab VaR and Risk Metrics Demo")
    print("=" * 70)

    # Generate sample return data
    np.random.seed(42)
    n_days = 1000
    mu_daily = 0.001     # 0.1% daily return
    sigma_daily = 0.02   # 2% daily volatility
    returns = np.random.normal(mu_daily, sigma_daily, n_days)

    print(f"\nSample Data: {n_days} daily returns")
    print(f"  Mean: {returns.mean():.6f}, Std: {returns.std():.6f}")
    print(f"  Annualized Mean: {returns.mean()*252:.4f} ({returns.mean()*252*100:.2f}%)")
    print(f"  Annualized Vol:  {returns.std()*np.sqrt(252):.4f} ({returns.std()*np.sqrt(252)*100:.2f}%)")

    # 1. Value at Risk
    print("\n" + "-" * 50)
    print("1. Value at Risk (VaR)")
    print("-" * 50)

    for alpha in [0.01, 0.05, 0.10]:
        conf_level = (1 - alpha) * 100
        print(f"\n  Confidence Level: {conf_level:.0f}%")
        print(f"  {'-' * 40}")

        hv = historical_var(returns, alpha)
        pv = parametric_var(returns, alpha)
        mv = monte_carlo_var(1000000, mu_daily, sigma_daily,
                              n_sims=10000, alpha=alpha, horizon=1, seed=42)

        print(f"  Historical VaR:  {hv:.6f} ({hv*100:.2f}%)")
        print(f"  Parametric VaR:  {pv:.6f} ({pv*100:.2f}%)")
        print(f"  Monte Carlo VaR: {mv:.6f} ({mv*100:.2f}%)")

        # CVaR at same level
        cv = cvar(returns, alpha)
        print(f"  Conditional VaR: {cv:.6f} ({cv*100:.2f}%)")

    # 2. VaR Backtesting
    print("\n" + "-" * 50)
    print("2. VaR Backtesting")
    print("-" * 50)

    # Split data: use first half to compute VaR, second half to test
    train_returns = returns[:500]
    test_returns = returns[500:]

    # Rolling VaR forecasts (using expanding window)
    var_fcasts = np.zeros(len(test_returns))
    window = 252  # 1 year of daily data

    for t in range(len(test_returns)):
        if t < window:
            lookback = train_returns[-(window - t):]
        else:
            lookback = returns[500 + t - window:500 + t]
        var_fcasts[t] = parametric_var(lookback, alpha=0.05)

    bt_results = backtest_var(test_returns, var_fcasts, alpha=0.05)

    print(f"  Test period: {len(test_returns)} days")
    print(f"  Expected violations (5%): {bt_results['expected_exceedances']:.1f}")
    print(f"  Actual violations:        {bt_results['n_exceedances']}")
    print(f"  Violation rate:           {bt_results['violation_rate']:.4f} ({bt_results['violation_rate']*100:.2f}%)")
    print(f"  Kupiec POF stat:          {bt_results['kupiec_stat']:.4f}")
    print(f"  Kupiec p-value:           {bt_results['kupiec_pvalue']:.4f}")
    print(f"  Christoffersen stat:      {bt_results['christoffersen_stat']:.4f}")
    print(f"  Christoffersen p-value:   {bt_results['christoffersen_pvalue']:.4f}")

    if bt_results['kupiec_pvalue'] is not None and not np.isnan(bt_results['kupiec_pvalue']):
        if bt_results['kupiec_pvalue'] > 0.05:
            print("  => VaR model appears well-calibrated (fail to reject at 5%)")
        else:
            print("  => VaR model may be mis-calibrated (reject at 5%)")

    # 3. Risk Metrics
    print("\n" + "-" * 50)
    print("3. Performance Risk Metrics")
    print("-" * 50)

    # Reconstruct price series
    prices = 100 * np.cumprod(1.0 + returns)
    benchmark_returns = np.random.normal(0.0008, 0.015, n_days)

    rf_annual = 0.02  # 2% risk-free rate

    sr = sharpe_ratio(returns, rf=rf_annual, periods_per_year=252)
    sort = sortino_ratio(returns, rf=rf_annual, target=0.0, periods_per_year=252)
    mdd = max_drawdown(prices)
    cal = calmar_ratio(returns, periods_per_year=252)
    ir = information_ratio(returns, benchmark_returns, periods_per_year=252)

    print(f"  Sharpe Ratio:      {sr:.4f}")
    print(f"  Sortino Ratio:     {sort:.4f}")
    print(f"  Max Drawdown:      {mdd:.4f} ({mdd*100:.2f}%)")
    print(f"  Calmar Ratio:      {cal:.4f}")
    print(f"  Information Ratio: {ir:.4f}")

    # Summary table
    print("\n" + "=" * 70)
    print("Risk Summary")
    print("=" * 70)
    print(f"  {'Metric':<25} {'Value':<15}")
    print(f"  {'-'*25} {'-'*15}")
    print(f"  {'95% Historical VaR':<25} {historical_var(returns, 0.05):.6f}")
    print(f"  {'95% Parametric VaR':<25} {parametric_var(returns, 0.05):.6f}")
    print(f"  {'95% CVaR':<25} {cvar(returns, 0.05):.6f}")
    print(f"  {'Annualized Sharpe':<25} {sr:.4f}")
    print(f"  {'Annualized Sortino':<25} {sort:.4f}")
    print(f"  {'Max Drawdown':<25} {mdd:.4f}")
    print(f"  {'Calmar Ratio':<25} {cal:.4f}")


if __name__ == '__main__':
    main()
