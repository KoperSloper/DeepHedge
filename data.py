import jax
import jax.numpy as jnp
from dataclasses import dataclass
from utils import simulate_gbm_paths

@dataclass
class MarketConfig:
    """Configuration for the Market Environment"""
    S0: float = 100.0       # Initial Stock Price
    K: float = 100.0        # Strike Price
    mu: float = 0.08         # Drift (Risk-neutral usually 0)
    sigma: float = 0.2      # Volatility (20%)
    T: float = 30/365.0     # Time to maturity (30 days)
    n_steps: int = 30       # Rebalancing frequency (Daily)
    
    # Dataset Sizes
    n_train: int = 100_000  # Number of paths for training
    n_val: int = 10_000     # Number of paths for validation
    n_test: int = 20_000    # Number of paths for final testing

def generate_market_features(prices, T, K):
    """
    Constructs the static features for the Neural Network:
    1. Log-Moneyness: ln(S_t / K)
    2. Time-to-Maturity: (T - t) / T  [Normalized to 1.0 -> 0.0]
    
    Args:
        prices: Array of shape (n_paths, n_steps + 1)
        T: Total time to maturity
        K: Strike price
        
    Returns:
        jnp.ndarray: Feature tensor of shape (n_paths, n_steps + 1, 2)
                     Feature 0: Log-Moneyness
                     Feature 1: Normalized Time
    """
    n_paths, n_timesteps = prices.shape
    
    log_moneyness = jnp.log(prices / K)

    time_grid = jnp.linspace(T, 0, n_timesteps)
    norm_ttm = time_grid / T
    
    norm_ttm_batch = jnp.tile(norm_ttm, (n_paths, 1))

    # Shape: (n_paths, n_steps + 1, 2)
    features = jnp.stack([log_moneyness, norm_ttm_batch], axis=-1)
    
    return features

def get_datasets(master_key, config: MarketConfig):
    """
    Generates Train, Validation, and Test sets containing Prices and Features.
    
    Returns a dictionary with keys 'train', 'val', 'test'.
    Each value is a tuple: (prices, features)
    """
    # Split the random key into 3 sub-keys
    k1, k2, k3 = jax.random.split(master_key, 3)
    
    datasets = {}
    
    # Loop to create each set
    for mode, key, count in [('train', k1, config.n_train), 
                             ('val',   k2, config.n_val), 
                             ('test',  k3, config.n_test)]:
        
        # 1. Simulate Raw Prices
        # Shape: (count, n_steps + 1)
        prices = simulate_gbm_paths(
            key, 
            config.S0, 
            config.mu, 
            config.sigma, 
            config.T, 
            config.n_steps, 
            count
        )
        
        # 2. Build Static Features
        # Shape: (count, n_steps + 1, 2)
        features = generate_market_features(prices, config.T, config.K)
        
        datasets[mode] = (prices, features)
        
        print(f"Generated {mode.upper()} set: {count} paths.")
        
    return datasets