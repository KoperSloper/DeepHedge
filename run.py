import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os

# Ensure these match your local file structure
from data import MarketConfig, get_datasets
from policy import DeepHedgingPolicy
# Updated imports to include save/load
from trainer import create_train_state, train_step, eval_step, save_params, load_params
from utils import entropic_loss, calculate_indifference_price

def main():
    seed = 42
    master_key = jax.random.PRNGKey(seed)
    
    config = MarketConfig(
        S0=100.0, K=100.0, mu=0.05, sigma=0.2, T=30/365.0, n_steps=30,
        n_train=50_000, n_val=5000, n_test=10_000
    )
    
    BATCH_SIZE = 256
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    RISK_AVERSION = 1.0   
    COST_LAMBDA = 0.005   
    
    PATIENCE = 30            
    IMPROVEMENT_RATIO = 0.99 
    MODEL_FILENAME = "best_hedging_model.msgpack"

    print("--- 1. Generating Data ---")
    data = get_datasets(master_key, config)
    train_prices, train_feats = data['train']
    val_prices, val_feats = data['val']
    test_prices, test_feats = data['test']
    
    train_feats_dec = train_feats[:, :-1, :]
    val_feats_dec = val_feats[:, :-1, :]
    test_feats_dec = test_feats[:, :-1, :]

    print("--- 2. Initializing Model ---")
    model = DeepHedgingPolicy(
        hidden_layer_sizes=[64, 64], 
        output_dim=1
    )
    
    rng_init, rng_train = jax.random.split(master_key)
    state = create_train_state(model, rng_init, LEARNING_RATE, (1, 3))
    
    def payoff_fn(prices):
        S_T = prices[:, -1]
        return jnp.maximum(S_T - config.K, 0.0)

    print(f"--- 3. Starting Training ({EPOCHS} Epochs) ---")
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    n_batches = train_prices.shape[0] // BATCH_SIZE
    
    for epoch in range(EPOCHS):
        rng_train, rng_shuffle = jax.random.split(rng_train)
        perms = jax.random.permutation(rng_shuffle, train_prices.shape[0])
        perms = perms[:n_batches * BATCH_SIZE]
        perms = perms.reshape((n_batches, BATCH_SIZE))
        
        batch_loss_sum = 0
        
        with tqdm(total=n_batches, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch") as pbar:
            for i in range(n_batches):
                indices = perms[i]
                b_prices = train_prices[indices]
                b_feats = train_feats_dec[indices]
                
                state, loss = train_step(
                    state, b_prices, b_feats, payoff_fn, COST_LAMBDA, RISK_AVERSION
                )
                
                if jnp.isnan(loss):
                    print(f"\n[Error] NaN Loss detected at Epoch {epoch+1}, Batch {i}")
                    sys.exit(1)

                batch_loss_sum += loss
                pbar.set_postfix({"loss": f"{loss:.4f}"})
                pbar.update(1)
        
        avg_train_loss = batch_loss_sum / n_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        val_Y = eval_step(state, val_prices, val_feats_dec, payoff_fn, COST_LAMBDA)
        val_loss = entropic_loss(val_Y, RISK_AVERSION)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.5f} | Val Loss: {val_loss:.5f}")

        # --- EARLY STOPPING LOGIC ---
        # "If val loss_t+1 is not smaller than 0.99 x val_loss_t increase patience"
        target_to_beat = best_val_loss * IMPROVEMENT_RATIO
        
        if val_loss < target_to_beat:
            print(f" >> Improvement! ({best_val_loss:.4f} -> {val_loss:.4f}). Saving model...")
            best_val_loss = val_loss
            patience_counter = 0
            # SAVE THE MODEL
            save_params(state, MODEL_FILENAME)
        else:
            patience_counter += 1
            print(f" >> No significant improvement. Patience: {patience_counter}/{PATIENCE}")
            
        if patience_counter >= PATIENCE:
            print(f"\n[Early Stopping] No improvement for {PATIENCE} epochs. Stopping.")
            break

    print("\n--- 4. Final Evaluation ---")
    
    # RELOAD THE BEST MODEL
    if os.path.exists(MODEL_FILENAME):
        print(f"Reloading best model from {MODEL_FILENAME}...")
        state = load_params(state, MODEL_FILENAME)
    else:
        print("Warning: No saved model found (did training fail?). Using last state.")
    
    test_Y = eval_step(state, test_prices, test_feats_dec, payoff_fn, COST_LAMBDA)
    final_test_loss = entropic_loss(test_Y, RISK_AVERSION)
    price_p0 = calculate_indifference_price(final_test_loss, RISK_AVERSION)
    bs_price_mc = jnp.mean(payoff_fn(test_prices))
    
    print(f"Deep Hedging Loss (Utility): {final_test_loss:.5f}")
    print(f"Indifference Price (p0):     {price_p0:.4f}")
    print(f"Premium Charged (p0 - RN):   {price_p0 - bs_price_mc:.4f}")
    
    # --- 5. Visualization ---
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title("Entropic Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    pnl_data = np.array(test_Y) + float(price_p0)
    plt.hist(pnl_data, bins=50, alpha=0.7, color='blue', edgecolor='black', label='Deep Hedge')
    plt.axvline(0, color='red', linestyle='--')
    plt.title(f"Final P&L Distribution (Charged {price_p0:.2f})")
    plt.xlabel("Profit / Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("training_results_early_stop.png")
    print("Results saved to training_results_early_stop.png")

if __name__ == "__main__":
    main()