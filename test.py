import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from data import MarketConfig, get_datasets
from policy import DeepHedgingPolicy
from trainer import create_train_state, load_params, eval_step
from utils import entropic_loss, calculate_indifference_price, compute_pnl
from jax.scipy.stats import norm

def get_bs_greeks(S, K, T_remaining, r, sigma):
    """Computes BS Delta and Gamma."""
    T_safe = jnp.maximum(T_remaining, 1e-5)
    d1 = (jnp.log(S / K) + (r + 0.5 * sigma**2) * T_safe) / (sigma * jnp.sqrt(T_safe))
    
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * jnp.sqrt(T_safe))
    
    return delta, gamma

def strategy_black_scholes(prices, K, T, r, sigma):
    """Naive Black-Scholes Delta Hedging."""
    n_paths, n_steps_plus_1 = prices.shape
    n_steps = n_steps_plus_1 - 1
    dt = T / n_steps
    
    deltas = []
    
    for t in range(n_steps):
        t_rem = T - t * dt
        S_t = prices[:, t]
        d_t, _ = get_bs_greeks(S_t, K, t_rem, r, sigma)
        deltas.append(d_t)
        
    return jnp.stack(deltas, axis=1)

def strategy_whalley_wilmott(prices, K, T, r, sigma, cost_lambda, risk_aversion):
    """Whalley-Wilmott Asymptotic Hedging (No-Transaction Band)."""
    n_paths, n_steps_plus_1 = prices.shape
    n_steps = n_steps_plus_1 - 1
    dt = T / n_steps
    
    deltas = []
    prev_pos = jnp.zeros(n_paths)
    
    for t in range(n_steps):
        t_rem = T - t * dt
        S_t = prices[:, t]
        
        bs_delta, gamma = get_bs_greeks(S_t, K, t_rem, r, sigma)
        
        # Calculate Band Width (H)
        # H = ( (3/2) * (lambda * S * Gamma^2) / Risk_Aversion )^(1/3)
        numer = 1.5 * cost_lambda * S_t * (gamma ** 2)
        denom = risk_aversion + 1e-6
        bandwidth = (numer / denom) ** (1.0/3.0)
        
        upper_bound = bs_delta + bandwidth
        lower_bound = bs_delta - bandwidth
        
        new_pos = jnp.clip(prev_pos, lower_bound, upper_bound)
        deltas.append(new_pos)
        prev_pos = new_pos
        
    return jnp.stack(deltas, axis=1)


def main():
    print("--- 1. Configuration & Data ---")
    # Must match training config for valid comparison
    config = MarketConfig(
        S0=100.0, K=100.0, mu=0.05, sigma=0.2, T=30/365.0, n_steps=30,
        n_train=100, n_val=100, n_test=10_000 # Only n_test matters here
    )
    
    COST_LAMBDA = 0.005
    RISK_AVERSION = 1.0
    MODEL_FILENAME = "best_hedging_model.msgpack"
    
    if not os.path.exists(MODEL_FILENAME):
        print(f"Error: {MODEL_FILENAME} not found. Run run.py first.")
        sys.exit(1)

    seed = 123 # Different seed from training
    master_key = jax.random.PRNGKey(seed)
    data = get_datasets(master_key, config)
    test_prices, test_feats = data['test']
    test_feats_dec = test_feats[:, :-1, :] # Slice for policy input
    
    # Payoff (Call Option)
    payoff_fn = lambda p: jnp.maximum(p[:, -1] - config.K, 0.0)
    payoffs = payoff_fn(test_prices)
    bs_price_mc = jnp.mean(payoffs)

    print(f"\nTest Set Size: {test_prices.shape[0]} paths")
    print(f"Risk Neutral Price (MC): {bs_price_mc:.4f}")

    print("\n--- 2. Evaluating Deep Hedging ---")
    model = DeepHedgingPolicy(hidden_layer_sizes=[64, 64], output_dim=1)
    
    # Init empty state and load parameters
    rng_init = jax.random.PRNGKey(0)
    state = create_train_state(model, rng_init, 0.0, (1, 3))
    state = load_params(state, MODEL_FILENAME)
    
    dh_pnl = eval_step(state, test_prices, test_feats_dec, payoff_fn, COST_LAMBDA)
    dh_loss = entropic_loss(dh_pnl, RISK_AVERSION)
    dh_price = calculate_indifference_price(dh_loss, RISK_AVERSION)
    
    print(f"Deep Hedge Price: {dh_price:.4f}")
    print(f"Premium Charged:  {dh_price - bs_price_mc:.4f}")

    print("\n--- 3. Evaluating Benchmarks ---")
    
    # Benchmark A: Naive Black-Scholes
    bs_deltas = strategy_black_scholes(test_prices, config.K, config.T, 0.0, config.sigma) # r=0 for BS formula
    bs_pnl_vec = compute_pnl(test_prices, bs_deltas, payoffs, COST_LAMBDA)
    bs_loss = entropic_loss(bs_pnl_vec, RISK_AVERSION)
    bs_bench_price = calculate_indifference_price(bs_loss, RISK_AVERSION)
    
    # Benchmark B: Whalley-Wilmott
    ww_deltas = strategy_whalley_wilmott(test_prices, config.K, config.T, 0.0, config.sigma, COST_LAMBDA, RISK_AVERSION)
    ww_pnl_vec = compute_pnl(test_prices, ww_deltas, payoffs, COST_LAMBDA)
    ww_loss = entropic_loss(ww_pnl_vec, RISK_AVERSION)
    ww_bench_price = calculate_indifference_price(ww_loss, RISK_AVERSION)

    print(f"{'Strategy':<20} | {'Indiff. Price':<15} | {'Premium':<10} | {'Utility Loss':<10}")
    print("-" * 65)
    print(f"{'Deep Hedging':<20} | {dh_price:<15.4f} | {dh_price - bs_price_mc:<10.4f} | {dh_loss:.4f}")
    print(f"{'Whalley-Wilmott':<20} | {ww_bench_price:<15.4f} | {ww_bench_price - bs_price_mc:<10.4f} | {ww_loss:.4f}")
    print(f"{'Naive BS':<20} | {bs_bench_price:<15.4f} | {bs_bench_price - bs_price_mc:<10.4f} | {bs_loss:.4f}")

    plt.figure(figsize=(10, 6))
    
    # Add back the premium to center distributions around 0 for visual comparison
    plt.hist(dh_pnl + dh_price, bins=50, alpha=0.5, label='Deep Hedging', density=True, color='blue')
    plt.hist(ww_pnl_vec + ww_bench_price, bins=50, alpha=0.5, label='Whalley-Wilmott', density=True, color='pink')
    plt.hist(bs_pnl_vec + bs_bench_price, bins=50, alpha=0.3, label='Naive BS', density=True, color='gray')
    
    plt.axvline(0, color='black', linestyle='--', alpha=0.5)
    plt.title("P&L Distribution Comparison (Centered)")
    plt.xlabel("Profit / Loss (after charging premium)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("benchmark_comparison.png")
    print("\nSaved comparison plot to benchmark_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()