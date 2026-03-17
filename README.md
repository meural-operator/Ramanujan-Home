# Ramanujan Machine Engine V2

The Ramanujan Machine is an algorithmic approach to discover new mathematical conjectures. 

This version (V2) introduces a major architectural shift toward **AI-Guided Discovery**, utilizing Deep Reinforcement Learning and high-performance GPU acceleration to explore mathematical spaces that were previously computationally unreachable.

## 🚀 Key Features (Engine V2)

### 1. High-Performance GPU Engine
*   **Vectorized CUDA Evaluation**: 20M+ GCF evaluations per second on standard NVIDIA hardware (RTX 4000 Ada).
*   **Zero-Latency Matching**: Hash-matching occurs entirely on-GPU using `torch.isin`, eliminating the Python bottleneck.
*   **Asynchronous Pipeline**: A multi-threaded architecture allows the GPU to explore while a background CPU thread verifies preliminary hits in real-time.

### 2. Deep RL Framework (`ramanujan/math_ai/`)
*   **Actor-Critic Models**: Neural networks that learn to predict the most promising directions for mathematical convergence.
*   **Neural-Guided MCTS**: An AlphaGo-style search (`NeuralMCTSPolyDomain`) that prunes the search space based on deep learning "rollouts" before passing it to the GPU.
*   **Modular Architecture**: Plug-and-play support for custom RL agents, environments, and reward functions.

## Installation

Clone the repo and install dependencies.
```bash
pip install -e .
pip install torch tqdm mpmath
```

## Quick Start: Running the Search

To start a new discovery execution (e.g., for the Euler-Mascheroni constant):

```python
from ramanujan.LHSHashTable import LHSHashTable
from ramanujan.enumerators.GPUEfficientGCFEnumerator import GPUEfficientGCFEnumerator
from ramanujan.poly_domains.NeuralMCTSPolyDomain import NeuralMCTSPolyDomain
from ramanujan.constants import g_const_dict

# 1. Build LHS Table
const_val = g_const_dict['euler-mascheroni']
lhs = LHSHashTable('euler_mascheroni.db', 30, [const_val])

# 2. Setup AI-Guided Search Domain
poly_search_domain = NeuralMCTSPolyDomain(
    a_deg=3, a_coef_range=[-20, 20],
    b_deg=3, b_coef_range=[-20, 20],
    target_val=const_val,
    mcts_simulations=1000
)

# 3. Execute High-Performance Parallel Enumeration
enumerator = GPUEfficientGCFEnumerator(lhs, poly_search_domain, [const_val])
results = enumerator.full_execution()

# 4. Print Results
enumerator.print_results(results)
```

## Architectural Breakdown

### `math_ai/Agents`
Contains the RL logic. The `AlphaTensorMCTS` agent uses the Actor-Critic network's Upper Confidence Bound (UCB) predictions to navigate the polynomial coefficient space.

### `enumerators/GPUEfficientGCFEnumerator`
The high-speed backbone. It manages:
1.  **Main Thread**: Feeds tensor chunks to CUDA.
2.  **Worker Thread**: Consumes a `queue.Queue` of hits and performs `mpmath` high-precision verification asynchronously.
3.  **Dual-TQDM**: Visualizes both GPU throughput and CPU verification progress simultaneously.

---
For more information, please visit [RamanujanMachine.com](https://www.RamanujanMachine.com).

