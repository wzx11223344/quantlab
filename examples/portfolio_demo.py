"""
Portfolio Optimization Demo

Demonstrates Markowitz efficient frontier, min variance, max Sharpe,
Black-Litterman, Risk Parity, and HRP portfolios.

Run: python examples/portfolio_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from quantlab.portfolio import (
    efficient_frontier,
    min_variance_portfolio,
    max_sharpe_portfolio,
    tangency_portfolio,
    black_litterman,
    risk_parity_portfolio,
    hrp,
)


def main():
    print("=" * 70)
    print("QuantLab Portfolio Optimization Demo")
    print("=" * 70)

    # Sample data: 5 assets
    expected_returns = np.array([0.10, 0.12, 0.08, 0.15, 0.07])
    cov_matrix = np.array([
        [0.040, 0.010, 0.005, 0.008, 0.003],
        [0.010, 0.090, 0.020, 0.012, 0.005],
        [0.005, 0.020, 0.030, 0.006, 0.004],
        [0.008, 0.012, 0.006, 0.100, 0.002],
        [0.003, 0.005, 0.004, 0.002, 0.025],
    ])
    asset_names = ['Asset A', 'Asset B', 'Asset C', 'Asset D', 'Asset E']
    rf = 0.03  # 3% risk-free rate

    print(f"\nAssets: {asset_names}")
    print(f"Expected Returns: {expected_returns}")
    print(f"Risk-free Rate: {rf}")
    print(f"Covariance Matrix:\n{cov_matrix}")

    # 1. Minimum Variance Portfolio
    print("\n" + "-" * 50)
    print("1. Global Minimum Variance Portfolio (GMVP)")
    print("-" * 50)

    w_gmv = min_variance_portfolio(cov_matrix)
    port_ret = w_gmv @ expected_returns
    port_vol = np.sqrt(w_gmv @ cov_matrix @ w_gmv)

    for i, (name, w) in enumerate(zip(asset_names, w_gmv)):
        print(f"  {name}: {w:.4f} ({w*100:.1f}%)")
    print(f"  Expected Return: {port_ret:.4f} ({port_ret*100:.2f}%)")
    print(f"  Volatility:      {port_vol:.4f} ({port_vol*100:.2f}%)")
    print(f"  Sharpe Ratio:    {(port_ret - rf) / port_vol:.4f}")

    # 2. Tangency / Max Sharpe Portfolio
    print("\n" + "-" * 50)
    print("2. Maximum Sharpe Ratio (Tangency) Portfolio")
    print("-" * 50)

    w_tan = max_sharpe_portfolio(expected_returns, cov_matrix, rf)
    port_ret_tan = w_tan @ expected_returns
    port_vol_tan = np.sqrt(w_tan @ cov_matrix @ w_tan)

    for i, (name, w) in enumerate(zip(asset_names, w_tan)):
        print(f"  {name}: {w:.4f} ({w*100:.1f}%)")
    print(f"  Expected Return: {port_ret_tan:.4f} ({port_ret_tan*100:.2f}%)")
    print(f"  Volatility:      {port_vol_tan:.4f} ({port_vol_tan*100:.2f}%)")
    print(f"  Sharpe Ratio:    {(port_ret_tan - rf) / port_vol_tan:.4f}")

    # 3. Efficient Frontier
    print("\n" + "-" * 50)
    print("3. Efficient Frontier (10 points)")
    print("-" * 50)

    rets, vols, weights = efficient_frontier(expected_returns, cov_matrix, n_points=10, rf=rf)
    print(f"  {'Point':<6} {'Return':<10} {'Volatility':<12} {'Weights'}")
    print(f"  {'-'*6} {'-'*10} {'-'*12} {'-'*30}")
    for i in range(0, len(rets), max(1, len(rets) // 10)):
        w_str = ', '.join([f'{w:.2f}' for w in weights[i]])
        print(f"  {i+1:<6} {rets[i]:.4f}      {vols[i]:.4f}        [{w_str}]")

    # 4. Black-Litterman
    print("\n" + "-" * 50)
    print("4. Black-Litterman Model")
    print("-" * 50)

    # Market equilibrium (prior) returns
    prior_returns = np.array([0.08, 0.06, 0.07, 0.09, 0.05])

    # Investor views
    # View 1: Asset A will outperform Asset B by 3%
    # View 2: Asset D will return 12%
    P = np.array([
        [1, -1, 0, 0, 0],   # A - B
        [0, 0, 0, 1, 0],    # D
    ])
    Q = np.array([0.03, 0.12])

    print(f"  Prior returns:      {prior_returns}")
    print(f"  View 1: A - B = {Q[0]:.2f}")
    print(f"  View 2: D = {Q[1]:.2f}")

    posterior_returns, posterior_cov = black_litterman(prior_returns, cov_matrix, P, Q)

    print(f"\n  Posterior returns:  {np.round(posterior_returns, 4)}")

    # BL-optimal portfolio
    w_bl = max_sharpe_portfolio(posterior_returns, posterior_cov, rf)
    print(f"\n  BL Max Sharpe weights:")
    for name, w in zip(asset_names, w_bl):
        print(f"    {name}: {w:.4f} ({w*100:.1f}%)")

    # 5. Risk Parity
    print("\n" + "-" * 50)
    print("5. Risk Parity (ERC) Portfolio")
    print("-" * 50)

    w_rp = risk_parity_portfolio(cov_matrix)
    port_vol_rp = np.sqrt(w_rp @ cov_matrix @ w_rp)
    mr = cov_matrix @ w_rp / port_vol_rp
    rc = w_rp * mr
    rc_pct = rc / rc.sum()

    for i, (name, w, rp) in enumerate(zip(asset_names, w_rp, rc_pct)):
        print(f"  {name}: {w:.4f} ({w*100:.1f}%), Risk Contrib: {rp*100:.1f}%")
    print(f"  Portfolio Vol: {port_vol_rp:.4f}")

    # 6. HRP
    print("\n" + "-" * 50)
    print("6. Hierarchical Risk Parity (HRP)")
    print("-" * 50)

    w_hrp = hrp(cov_matrix)
    for name, w in zip(asset_names, w_hrp):
        print(f"  {name}: {w:.4f} ({w*100:.1f}%)")

    # Summary
    print("\n" + "=" * 70)
    print("Portfolio Weights Summary")
    print("=" * 70)
    print(f"  {'Method':<20} {'A':>8} {'B':>8} {'C':>8} {'D':>8} {'E':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'Equal Weight':<20} {'0.2000':>8} {'0.2000':>8} {'0.2000':>8} {'0.2000':>8} {'0.2000':>8}")
    for method, w in [('Min Variance', w_gmv), ('Max Sharpe', w_tan),
                       ('Risk Parity', w_rp), ('HRP', w_hrp)]:
        print(f"  {method:<20} {w[0]:>8.4f} {w[1]:>8.4f} {w[2]:>8.4f} {w[3]:>8.4f} {w[4]:>8.4f}")


if __name__ == '__main__':
    main()
