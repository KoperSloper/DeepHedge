import jax
import jax.numpy as jnp
from flax.training import train_state
from flax import serialization
import optax
from functools import partial
import os

from utils import compute_pnl, entropic_loss

class TrainState(train_state.TrainState):
    pass

def create_train_state(model, key, learning_rate, input_shape):
    """Initializes the model, optimizer, and state."""
    dummy_input = jnp.ones(input_shape)
    variables = model.init(key, dummy_input)
    params = variables['params']
    tx = optax.adam(learning_rate)
    
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

def save_params(state, filename="best_model.msgpack"):
    """Saves the model parameters to a file."""
    with open(filename, "wb") as f:
        f.write(serialization.to_bytes(state.params))
    # print(f"Model saved to {filename}")

def load_params(state, filename="best_model.msgpack"):
    """Loads parameters from a file into the state."""
    with open(filename, "rb") as f:
        bytes_data = f.read()
    
    # We use the current state.params as a template structure
    new_params = serialization.from_bytes(state.params, bytes_data)
    return state.replace(params=new_params)

@partial(jax.jit, static_argnames=['payoff_fn'])
def train_step(state, batch_prices, batch_features, payoff_fn, cost_lambda, risk_aversion):
    
    def loss_fn(params):
        # Transpose to (Steps, Batch, Features) for scan
        scan_inputs = jnp.transpose(batch_features, (1, 0, 2))
        batch_size = scan_inputs.shape[1]
        init_carry = jnp.zeros((batch_size, 1))

        def scan_body(carry, x_t):
            prev_position = carry 
            model_input = jnp.concatenate([x_t, prev_position], axis=-1)
            
            delta_change = state.apply_fn({'params': params}, model_input)
            new_position = prev_position + delta_change
            
            return new_position, new_position

        _, deltas_stacked = jax.lax.scan(scan_body, init_carry, scan_inputs)
        
        deltas = jnp.transpose(deltas_stacked, (1, 0, 2))
        deltas = jnp.squeeze(deltas, axis=-1)
        
        payoff = payoff_fn(batch_prices)
        Y = compute_pnl(batch_prices, deltas, payoff, cost_lambda)
        loss = entropic_loss(Y, risk_aversion)
        
        return loss

    loss_val, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    
    return new_state, loss_val

@partial(jax.jit, static_argnames=['payoff_fn'])
def eval_step(state, batch_prices, batch_features, payoff_fn, cost_lambda):
    scan_inputs = jnp.transpose(batch_features, (1, 0, 2))
    batch_size = scan_inputs.shape[1]
    init_carry = jnp.zeros((batch_size, 1))

    def scan_body_eval(carry, x_t):
        prev_position = carry
        model_input = jnp.concatenate([x_t, prev_position], axis=-1)
        
        delta_change = state.apply_fn({'params': state.params}, model_input)
        
        new_position = prev_position + delta_change
        return new_position, new_position

    _, deltas_stacked = jax.lax.scan(scan_body_eval, init_carry, scan_inputs)
    
    deltas = jnp.transpose(deltas_stacked, (1, 0, 2))
    deltas = jnp.squeeze(deltas, axis=-1)
    
    payoff = payoff_fn(batch_prices)
    Y = compute_pnl(batch_prices, deltas, payoff, cost_lambda)
    
    return Y