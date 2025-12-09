import jax
import jax.numpy as jnp

def simulate_gbm_paths(key, S0, mu, sigma, T, n_steps, n_paths):
    """
    Simulates Geometric Brownian Motion (GBM) paths using JAX.
    
    Formula: S_{t+1} = S_t * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
    
    Args:
        key: JAX random key (PRNGKey).
        S0: Initial stock price (float).
        mu: Drift (float). For risk-neutral hedging, this is usually r.
        sigma: Volatility (float).
        T: Time to maturity in years (float).
        n_steps: Number of time steps (int).
        n_paths: Number of paths to simulate (int).
        
    Returns:
        jnp.ndarray: Array of shape (n_paths, n_steps + 1) containing price paths.
    """
    dt = T / n_steps
    
    # Generate random Brownian increments (Z)
    # Shape: (n_paths, n_steps)
    Z = jax.random.normal(key, shape=(n_paths, n_steps))

    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * jnp.sqrt(dt) * Z
    
    # Calculate log-returns
    log_returns = drift + diffusion
    
    accumulated_log_returns = jnp.cumsum(log_returns, axis=1)
    accumulated_log_returns = jnp.concatenate(
        [jnp.zeros((n_paths, 1)), accumulated_log_returns], axis=1
    )
    
    # Convert back to prices
    prices = S0 * jnp.exp(accumulated_log_returns)
    
    return prices

def compute_pnl(prices, deltas, payoff, cost_lambda=0.0):
    """
    Calculates the Net P&L (Y) for the portfolio.
    Y = Trading Gains - Transaction Costs - Option Payoff
    
    Args:
        prices: Share prices (n_paths, n_steps + 1).
        deltas: Hedging positions (n_paths, n_steps). 
                Note: delta[i] is held from t[i] to t[i+1].
        payoff: The liability Z to be paid at T (n_paths,).
        cost_lambda: Proportional transaction cost parameter (float).
        
    Returns:
        jnp.ndarray: The final P&L vector Y of shape (n_paths,).
    """
    # S_{t+1} - S_t
    price_changes = jnp.diff(prices, axis=1) # Shape: (n_paths, n_steps)
    
    # sum(delta_t * (S_{t+1} - S_t))
    trading_gains = jnp.sum(deltas * price_changes, axis=1)
            
    # Prepend 0 to deltas to calculate the first trade (0 to delta[0])
    # Shape becomes (n_paths, n_steps + 1)
    deltas_padded = jnp.concatenate([jnp.zeros((deltas.shape[0], 1)), deltas], axis=1)
    
    # Calculate change in position: delta_t - delta_{t-1}
    delta_changes = jnp.diff(deltas_padded, axis=1) # Shape: (n_paths, n_steps)

    costs = cost_lambda * prices[:, :-1] * jnp.abs(delta_changes)
    total_costs = jnp.sum(costs, axis=1)
    
    # Y = Gains - Costs - Payoff
    Y = trading_gains - total_costs - payoff
    
    return Y

def entropic_loss(Y, risk_aversion):
    """
    Computes the Exponential Utility Loss.
    
    Formula: E[exp(-lambda * Y)]
    """
    return jnp.mean(jnp.exp(-risk_aversion * Y))

def calculate_indifference_price(final_loss, risk_aversion):
    return (1.0 / risk_aversion) * jnp.log(final_loss)