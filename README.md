# Ramanujan Engine V3: Mathematical Discovery Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA_Ready-EE4C2C.svg)](https://pytorch.org/)
[![Distributed Compute](https://img.shields.io/badge/Computing-Distributed_Edge-yellow.svg)](https://firebase.google.com/)

A globally distributed, GPU-accelerated computing network bridging **Deep Reinforcement Learning (AlphaTensor MCTS)** with **PyTorch Tensor exhaustion** to mathematically discover novel continued fractions. Originating as a brute-force mathematical search, this repository has been heavily rewritten into a strictly-typed, plug-and-play **Modular Mathematical Framework**.

## 🌟 Key Modifications (V3 Architecture Refactor)
This repository represents a massive paradigm shift from the V2 legacy codebase:
1. **Abstract Interface Decoupling**: Decommissioned the monolithic hardcoded evaluator in favor of a universal 4-stage abstract plugin pipeline (`TargetConstant`, `BoundingStrategy`, `EnumeratorEngine`, `NetworkCoordinator`).
2. **Deep Reinforcement Bounds Pruning**: Integrated an advanced **AlphaTensor MCTS (Monte Carlo Tree Search)** heuristic that maps physical trajectories and intelligently slices coordinate spaces prior to exhausting GPU memory.
3. **Decentralized Zero-Loss Edge Nodes**: Stripped out legacy file-system tracking and deployed a hardened `sqlite3` edge cache that guarantees verified discoveries survive accidental power or network losses.
4. **Frictionless Distribution**: Volunteers worldwide can now join the compute cluster with a strictly autonomous 1-click `.bat` deployment that handles Micromamba isolation, dependencies, credential generation, and mathematical table initializations transparently.
5. **Research-Grade RL Training Suite**: Outfitted with a dedicated Curriculum Learning PPO cycle, asynchronous PyTorch scaling, and rigorous mathematical TensorBoard MLOps.

---

## 🏗️ Architecture & Hierarchy

The V3 engine is heavily decoupled using the **Adapter and Strategy Software Patterns**. The high-level execution flow is managed autonomously by the `V3PipelineExecutor`.

```mermaid
graph TD
    classDef core fill:#2d3436,stroke:#74b9ff,stroke-width:2px,color:#fff;
    classDef abstract fill:#0984e3,stroke:#fff,stroke-width:2px,color:#fff;
    classDef plugin fill:#00b894,stroke:#fff,stroke-width:2px,color:#fff;

    A[Network Coordinator API] ::: plugin -->|Delivers Boundary Space| B(V3PipelineExecutor) ::: core
    
    B -->|Hooks| C1(TargetConstant) ::: abstract
    B -->|Pipes raw limits| C2(BoundingStrategy) ::: abstract
    B -->|Sinks constrained space| C3(EnumeratorEngine) ::: abstract
    
    C1 -->|EulerMascheroniTarget| D1[Generates Precision Match Hash] ::: plugin
    C2 -->|MCTSStrategy| D2[Prunes Search Volume via Neural RL] ::: plugin
    C3 -->|CUDAEnumerator| D3[NVIDIA Tensor Accelerator] ::: plugin
    
    D3 -->|Yields Mathematical Hits| B
    B -->|Synchronizes Global DB| A
```

### Component Relationships
* **`TargetConstant`**: Injects absolute mathematical truth (e.g. `EulerMascheroniTarget`). Dictates floating-point threshold validations and generates required Look-Up-Tables (LHS keys).
* **`BoundingStrategy`**: Sequence of optimization filters. `MCTSStrategy` currently hooks up to external PyTorch weights (`em_mcts.pt`) to physically restrict impossible dimensions from being searched.
* **`EnumeratorEngine`**: The bare-metal exhaustive execution layer. `CUDAEnumerator` wraps the legacy 200-line highly optimized broadcasting Tensor matrices into a safe, encapsulated plugin.
* **`NetworkCoordinator`**: `FirebaseCoordinator` safely extracts REST and Pyrebase hooks to fetch distributed payloads from the centralized cloud architecture.

---

## 🗃️ Inventory of Problems & Features
The framework successfully generalizes the following domain complexities:
* **Euler-Mascheroni Constant Discovery**: Active high-precision execution domain attempting to find novel exact formulas mapping to `0.57721566...`
* **Neural Actor-Critic Trajectories**: `math_ai` directory actively hosts state spaces and physical environment transitions specific to polynomial continued fractions. 
* **Dynamic Grid Orchestration**: Synchronous and asynchronous mutually exclusive locks deployed via cloud infrastructure preventing thousands of GPU nodes from resolving overlapping blocks.

---

## 🚀 Execution Guide

### 1-Click Deployment (Windows Volunteers)
For individuals dedicating compute to the cluster:
1. `git clone https://github.com/meural-operator/ramanujan_engineV2.git`
2. `cd ramanujan_engineV2\ramanujan_client`
3. Execute **`run_client.bat`**
> *The script will natively secure an isolated Python 3.13 Micromamba container, automatically synthesize cloud credentials, seed the 30MB LHS lookup tables sequentially, and orbit your GPU into the live cluster.*

### Manual Research Execution (Scientists & Engineers)
**1. Environment Bootstrap:**
```bash
conda env create -f setup/environment.yml
# Or manually build Python 3.13 with PyTorch / MpMath
conda activate curiosity
```

**2. Seed Legacy Verification Math Tables (One-Time):**
```bash
python scripts/seed_euler_mascheroni_db.py
```

**3. Initialize Edge Evaluation Node Pipeline:**
```bash
cd ramanujan_client
python ramanujan_client.py
```

### Reinforcement Learning Module Training
To mature the MCTS node network on new constants or wider polynomial bounds:
```bash
cd research_training
python train.py --episodes 50000 --max-depth 200
```
> Track performance dynamically via: `tensorboard --logdir runs/`

---

## 📁 Repository Structure
```text
ramanujan_engineV2/
├── ramanujan/                 # Core Research Framework
│   ├── interfaces/            # V3 Abstract Base Classes (Contracts)
│   ├── constants/             # Target Mathematical Plugins
│   ├── enumerators/           # Hardware Acceleration Adapters
│   ├── strategies/            # Pipeline Pruning Plugins
│   ├── math_ai/               # Actor-Critic & AlphaTensor Brains
│   ├── poly_domains/          # Cartesian Space Cartography
│   └── coordinators/          # Network I/O Wrappers
│
├── ramanujan_client/          # Distributed Compute Node
│   ├── engine_bridge/         # Decoupled V3PipelineExecutor Router
│   ├── checkpoints/           # Compiled RL Weight Artifacts (.pt)
│   ├── ramanujan_client.py    # Autonomous Setup and Cycle Hook
│   └── run_client.bat         # Zero-Config Windows Deployer
│
├── research_training/         # PyTorch Dedicated Training Pipeline
│   ├── train.py               # PPO Curriculum Matrix
│   ├── config.yaml            # Hyperparameter Thresholds
│   └── eval_mcts.py           # Physical Node Visualizer
│
├── scripts/                   # Structural DB / Task Handlers
├── tests/                     # Strict Unit & Integration Protections
└── README.md                  # Documentation (You are here)
```
