# Deep Hedging with Transaction Costs (JAX/Flax)

![Project Status](https://img.shields.io/badge/Status-Experimental-orange)
![Framework](https://img.shields.io/badge/Built%20With-JAX%20%7C%20Flax-blue)

A Deep Learning project that (attempts) to solve the optimal hedging strategy for a European Call Option in the presence of proportional transaction costs. 

Unlike the classic Black-Scholes model which assumes frictionless trading, this agent learns to balance **risk reduction** against **transaction fees**. It autonomously discovers a "No-Transaction Band" strategy that outperforms traditional benchmarks.

## 📉 The Problem
In a frictionless world, you can perfectly hedge an option by continuously rebalancing to the Black-Scholes Delta ($\Delta_{BS}$). However, in the real world:
1.  **Trading costs money:** You pay a spread every time you buy or sell.
2.  **Continuous trading is impossible:** You can only trade at discrete steps.

If you blindly follow Black-Scholes, your P&L gets destroyed by transaction fees. This project uses a Neural Network to find the optimal "sweet spot" policy.

## 🧠 The Solution
We model the hedging task as an optimization problem under **Exponential Utility** (Entropic Loss).

* **Observation:** The agent sees its current position, the market Moneyness ($$\ln(S_t/K)$$), and the Time-to-Maturity.
* **Architecture:** An MLP built in **Flax** that processes the entire hedging timeline.
* **Policy:** Instead of outputting a raw position, the network outputs **Upper and Lower Bounds** ($b_l, b_u$). The agent only trades if the current position drifts outside these bounds.

$$\delta_{new} = \text{clamp}(\delta_{old}, b_l, b_u)$$

## 🛠️ Installation

This project requires Python 3.9+ and the JAX ecosystem.

```bash
# Clone the repository
git clone [https://github.com/KoperSloper/DeepHedge.git](https://github.com/KoperSloper/DeepHedge.git)
cd DeepHedge

# Install dependencies (JAX, Flax, Optax, Matplotlib, Tqdm)
pip install jax jaxlib flax optax matplotlib tqdm
