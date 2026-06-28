"""
Options Pricing Demo

Demonstrates all four option pricing methods available in QuantLab:
1. Black-Scholes closed form
2. CRR Binomial Tree
3. Monte Carlo simulation
4. Heston stochastic volatility

Run: python examples/options_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from quantlab.options import (
    black_scholes_price,
    black_scholes_greeks,
    implied_volatility,
    binomial_tree,
    mc_price,
    mc_price_asian,
    mc_price_barrier,
    heston_price,
)


def main():
    print("=" * 70)
    print("QuantLab Options Pricing Demo")
    print("=" * 70)

    # Parameters
    S = 100.0    # Spot price
    K = 105.0    # Strike
    T = 1.0      # 1 year
    r = 0.05     # 5% risk-free rate
    sigma = 0.2  # 20% volatility

    print(f"\nParameters: S={S}, K={K}, T={T}, r={r}, sigma={sigma}")

    # 1. Black-Scholes
    print("\n" + "-" * 50)
    print("1. Black-Scholes Pricing")
    print("-" * 50)

    call_price = black_scholes_price(S, K, T, r, sigma, 'call')
    put_price = black_scholes_price(S, K, T, r, sigma, 'put')
    print(f"  Call price: {call_price:.6f}")
    print(f"  Put price:  {put_price:.6f}")

    # Put-Call Parity check
    parity_lhs = call_price - put_price
    parity_rhs = S - K * np.exp(-r * T)
    print(f"  Put-Call Parity: C-P = {parity_lhs:.6f}, S-K*exp(-rT) = {parity_rhs:.6f}")

    # Greeks
    greeks = black_scholes_greeks(S, K, T, r, sigma)
    print("\n  Greeks:")
    print(f"  Delta (call): {greeks['delta_call']:.6f}")
    print(f"  Delta (put):  {greeks['delta_put']:.6f}")
    print(f"  Gamma:        {greeks['gamma']:.6f}")
    print(f"  Theta (call): {greeks['theta_call']:.6f}")
    print(f"  Vega:         {greeks['vega']:.6f}")
    print(f"  Rho (call):   {greeks['rho_call']:.6f}")

    # Implied Volatility
    iv = implied_volatility(call_price, S, K, T, r, 'call')
    print(f"\n  Implied Volatility from call price: {iv:.6f}")

    # 2. Binomial Tree
    print("\n" + "-" * 50)
    print("2. Binomial Tree (CRR)")
    print("-" * 50)

    for N in [10, 50, 100, 500]:
        euro = binomial_tree(S, K, T, r, sigma, N, 'call', 'european')
        amer = binomial_tree(S, K, T, r, sigma, N, 'call', 'american')
        print(f"  N={N:3d}: European={euro:.6f}, American={amer:.6f}")

    print(f"\n  Reference BS: {call_price:.6f}")

    # 3. Monte Carlo
    print("\n" + "-" * 50)
    print("3. Monte Carlo Simulation")
    print("-" * 50)

    seed = 42
    for n_sims in [10000, 50000, 100000]:
        price_mc, std_err = mc_price(S, K, T, r, sigma, n_sims, 'call',
                                      antithetic=True, control_variate=True, seed=seed)
        ci_low = price_mc - 1.96 * std_err
        ci_high = price_mc + 1.96 * std_err
        print(f"  n_sims={n_sims:6d}: price={price_mc:.6f}, std_err={std_err:.6f}")
        print(f"      95% CI: [{ci_low:.6f}, {ci_high:.6f}]")

    # Asian Option
    asian_price, asian_se = mc_price_asian(S, K, T, r, sigma, n_sims=50000, seed=seed)
    print(f"\n  Asian Call (arithmetic avg): {asian_price:.6f} (SE={asian_se:.6f})")

    # Barrier Option
    barrier = 120.0
    barrier_price, barrier_se = mc_price_barrier(
        S, K, T, r, sigma, barrier, n_sims=50000, barrier_type='up-and-out', seed=seed
    )
    print(f"  Up-and-Out Call (B={barrier}): {barrier_price:.6f} (SE={barrier_se:.6f})")

    # 4. Heston Model
    print("\n" + "-" * 50)
    print("4. Heston Stochastic Volatility")
    print("-" * 50)

    v0 = 0.04       # Initial variance
    kappa = 2.0     # Mean reversion speed
    theta = 0.04    # Long-run variance
    xi = 0.3        # Vol-of-vol
    rho = -0.5      # Correlation

    heston_call = heston_price(S, K, T, r, v0, kappa, theta, xi, rho, 'call')
    heston_put = heston_price(S, K, T, r, v0, kappa, theta, xi, rho, 'put')
    print(f"  Heston Call: {heston_call:.6f}")
    print(f"  Heston Put:  {heston_put:.6f}")
    print(f"  BS Call (sigma=sqrt(v0)=0.2): {call_price:.6f}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary of Call Option Prices (S=100, K=105, T=1, r=5%, sigma=20%)")
    print("=" * 70)
    print(f"  Black-Scholes (closed form): {call_price:.6f}")
    print(f"  Binomial Tree (N=100):       {binomial_tree(S, K, T, r, sigma, 100, 'call'):.6f}")
    price_mc, _ = mc_price(S, K, T, r, sigma, 100000, 'call', seed=seed)
    print(f"  Monte Carlo (100k paths):    {price_mc:.6f}")
    print(f"  Heston (v0=0.04):            {heston_call:.6f}")


if __name__ == '__main__':
    main()
