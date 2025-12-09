import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence

class DeepHedgingPolicy(nn.Module):
    hidden_layer_sizes: Sequence[int]  # e.g., [32, 32]
    output_dim: int = 1                # Number of assets to trade 

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim).
               The LAST column of x is 'prev_position'.
        
        Returns:
            delta_change: The adjustment to the position (delta_t - delta_{t-1}).
        """
        prev_position = x[:, -1:] 
        
        activation = x
        for size in self.hidden_layer_sizes:
            activation = nn.Dense(features=size)(activation)
        
            activation = nn.LayerNorm()(activation)
            
            activation = nn.relu(activation)
        
        lower_bound = nn.Dense(features=self.output_dim, name='head_lower')(activation)
        
        raw_width = nn.Dense(features=self.output_dim, name='head_width')(activation)
        band_width = nn.softplus(raw_width) 

        upper_bound = lower_bound + band_width

        beta = 20.0
        
        # Calculate excess/deficit with softplus
        excess = nn.softplus(beta * (prev_position - upper_bound)) / beta
        deficit = nn.softplus(beta * (lower_bound - prev_position)) / beta
        
        target_position = prev_position - excess + deficit
        
        delta_change = target_position - prev_position
        
        return delta_change