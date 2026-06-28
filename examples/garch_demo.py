"""
GARCH Volatility Modeling Demo

Demonstrates GARCH(1,1), GJR-GARCH, and EGARCH fitting and forecasting,
along with realized volatility estimators.

Run: python examples/garch_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from quantlab.volatility import (
    GARCH,
    GJR_GARCH,
    EGARCH,
    parkinson_volatility,
    garman_klass_volatility,
    yang_zhang_volatility,
)
from quantlab.utils import jarque_bera, log_returns


def main():
    print("=" * 70)
    print("QuantLab GARCH Volatility Modeling Demo")
    print("=" * 70)

    # Generate synthetic returns with volatility clustering
    np.random.seed(42)
    n_days = 1000

    # Simulate GARCH(1,1) process
    omega_true = 0.00001
    alpha_true = 0.10
    beta_true = 0.85

    returns = np.zeros(n_days)
    sigma2 = np.zeros(n_days)
    sigma2[0] = omega_true / (1 - alpha_true - beta_true)

    for t in range(1, n_days):
        sigma2[t] = omega_true + alpha_true * returns[t-1]**2 + beta_true * sigma2[t-1]
        returns[t] = np.sqrt(sigma2[t]) * np.random.standard_normal()

    print(f"\nSample: {n_days} daily returns (decimal)")
    print(f"  Mean:   {returns.mean()*100:.4f}%")
    print(f"  Std:    {returns.std()*100:.4f}%")
    print(f"  Skew:   {np.mean((returns - returns.mean())**3) / returns.std()**3:.4f}")
    print(f"  Kurt:   {np.mean((returns - returns.mean())**4) / returns.std()**4:.4f}")

    # Normality test
    jb_stat, jb_pval = jarque_bera(returns)
    print(f"  Jarque-Bera: {jb_stat:.2f} (p={jb_pval:.4f})")

    # 1. GARCH(1,1)
    print("\n" + "-" * 50)
    print("1. GARCH(1,1) Model")
    print("-" * 50)

    garch = GARCH(p=1, q=1)
    garch.fit(returns, verbose=False)
    omega, alpha, beta = garch.params
    persistence = alpha + beta

    print(f"  True params:    omega={omega_true*10000:.4f}e-4, alpha={alpha_true:.4f}, beta={beta_true:.4f}")
    print(f"  Fitted params:  omega={omega:.6f}, alpha={alpha:.4f}, beta={beta:.4f}")
    print(f"  Persistence:    {persistence:.4f}")
    print(f"  Uncond. vol:    {np.sqrt(omega/(1-persistence))*np.sqrt(252):.4f}% annualized")

    # Forecast
    horizon = 20
    forecast = garch.forecast(horizon)
    print(f"\n  {horizon}-day volatility forecast (daily, %):")
    for h in [0, 4, 9, 19]:
        annual = np.sqrt(forecast[h] * 252) * 100
        print(f"    Day {h+1:2d}: {np.sqrt(forecast[h])*100:.4f}% (annualized: {annual:.2f}%)")

    # 2. GJR-GARCH
    print("\n" + "-" * 50)
    print("2. GJR-GARCH (Leverage Effect)")
    print("-" * 50)

    gjr = GJR_GARCH()
    gjr.fit(returns, verbose=False)
    omega_g, alpha_g, gamma_g, beta_g = gjr.params
    persistence_g = alpha_g + 0.5 * gamma_g + beta_g

    print(f"  omega={omega_g:.6f}, alpha={alpha_g:.4f}, gamma={gamma_g:.4f}, beta={beta_g:.4f}")
    print(f"  Leverage parameter gamma: {gamma_g:.4f}")
    if gamma_g > 0:
        print("  => Negative shocks increase volatility more than positive shocks")
    print(f"  Persistence: {persistence_g:.4f}")

    # 3. EGARCH
    print("\n" + "-" * 50)
    print("3. EGARCH Model")
    print("-" * 50)

    egarch = EGARCH()
    egarch.fit(returns, verbose=False)
    omega_e, alpha_e, gamma_e, beta_e = egarch.params

    print(f"  omega={omega_e:.6f}, alpha={alpha_e:.4f}, gamma={gamma_e:.4f}, beta={beta_e:.4f}")
    if gamma_e < 0:
        print("  => Negative gamma confirms leverage effect (asymmetric volatility)")

    # 4. Realized Volatility Estimators
    print("\n" + "-" * 50)
    print("4. Realized Volatility Estimators")
    print("-" * 50)

    # Simulate OHLC data from a GBM
    np.random.seed(123)
    n = 252
    S0 = 100.0
    mu = 0.05
    sigma_true = 0.20

    dt = 1.0 / 252
    Z = np.random.standard_normal(n)
    log_prices = np.log(S0) + np.cumsum((mu - 0.5 * sigma_true**2) * dt + sigma_true * np.sqrt(dt) * Z)
    close = np.exp(log_prices)

    # Simulate OHLC from close
    daily_vol = sigma_true * np.sqrt(dt)
    open_p = close * np.exp(daily_vol * np.random.standard_normal(n) * 0.3)
    intraday_range = daily_vol * np.abs(np.random.standard_normal(n))
    high = np.maximum(open_p, close) * (1.0 + intraday_range * 0.5)
    low = np.minimum(open_p, close) * (1.0 - intraday_range * 0.5)

    close_to_close_vol = np.std(np.diff(np.log(close))) * np.sqrt(252)
    park_vol = parkinson_volatility(high, low)
    gk_vol = garman_klass_volatility(open_p, high, low, close)
    yz_vol = yang_zhang_volatility(open_p, high, low, close)

    print(f"  True volatility:  {sigma_true*100:.1f}%")
    print(f"  Close-to-Close:   {close_to_close_vol*100:.2f}%")
    print(f"  Parkinson:        {park_vol*100:.2f}%")
    print(f"  Garman-Klass:     {gk_vol*100:.2f}%")
    print(f"  Yang-Zhang:       {yz_vol*100:.2f}%")

    # Summary
    print("\n" + "=" * 70)
    print("Volatility Summary")
    print("=" * 70)
    print(f"  {'Model':<25} {'Persist':<10} {'Ann. Vol':<12} {'Uncond. Vol':<15}")
    print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*15}")
    unc_v = np.sqrt(omega / (1 - persistence)) * np.sqrt(252) * 100 if persistence < 1.0 else 0
    print(f"  {'GARCH(1,1)':<25} {persistence:<10.4f} {np.sqrt(garch.fitted_var[-1])*np.sqrt(252)*100:<12.2f}% {unc_v:<15.2f}%")

    unc_vg = np.sqrt(omega_g / (1 - persistence_g)) * np.sqrt(252) * 100 if persistence_g < 1.0 else 0
    print(f"  {'GJR-GARCH':<25} {persistence_g:<10.4f} {np.sqrt(gjr.fitted_var[-1])*np.sqrt(252)*100:<12.2f}% {unc_vg:<15.2f}%")


if __name__ == '__main__':
    main()
