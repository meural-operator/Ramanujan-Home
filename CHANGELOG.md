# Ramanujan Machine Upgrades (Branch: experimental-ai-gpu)

## Overview
Successfully integrated a suite of modern high-performance computational paradigms and artificial intelligence routines into the Ramanujan Machine to expand integer relation discovery methodologies beyond discrete combinatorial searches. 

All logic dynamically preserves backward compatibility and runs independently via the `experimental-ai-gpu` branch on PyTorch 2.10 under the internal `curiosity` Conda environment.

## 1. Computation & Hardware Acceleration (GPU Tensorization)
- **`ramanujan/enumerators/GPUEfficientGCFEnumerator.py`**: [NEW] Implemented a PyTorch GPU-batched enumerator capable of parallelizing convergent series evaluations across massive polynomial arrays, replacing multi-processing loop overheads.
- **`ramanujan/enumerators/__init__.py`**: [MODIFY] Exported `GPUEfficientGCFEnumerator`.
- **`ramanujan/constants.py`**: [MODIFY] Refactored legacy `sympy.core.compatibility.with_metaclass` usage to standard Python 3.x metaclass mappings, enabling safe compatibility for deep learning module integrations.

## 2. Mathematical Pre-Filtering & Asymptotic Limits
- **`ramanujan/utils/asymptotic_filter.py`**: [NEW] Encodes an $O(1)$ algebraic check utilizing Pringsheim/Worpitzky bounding limits to logically purge numerically divergent combinations strictly, avoiding GPU processing completely.
- **`ramanujan/enumerators/FREnumerator.py`**: [MODIFY] Expanded the internal LLM-PSLQ sub-routine to extrapolate 1-dimensional constant targets dynamically into massive quadratic N-dimensional spaces ($X \cdot Y$, $X^2$) discovering cross-term convergences.

## 3. Heuristic Domain Search Strategies
Instead of relying strictly on grid combinatorics via the exhaustive iterations:
- **`ramanujan/poly_domains/CartesianProductPolyDomain.py`**: [MODIFY] Applied soft deprecation flags signaling limits traversing higher dimensions.
- **`ramanujan/poly_domains/MCTSPolyDomain.py`**: [NEW] Maps coefficients into parallel Monte Carlo rollouts to logically structure bounding limits isolating numerical sequence targets dynamically.
- **`ramanujan/poly_domains/ContinuousRelaxationDomain.py`**: [NEW] Replaces the discrete search arrays with a continuous `torch.optim.Adam` logic flow, optimizing boundaries based on fractional gradients descending toward tight numerical convergence constraints.

## 4. Deep Learning & Symbolic Intelligence Framework
Constructed the baseline architecture parameters in `<ramanujan/ai/>` facilitating autonomous parameter convergence trajectories:
- **`ramanujan/ai/symbolic_regression.py`**: [NEW] Encapsulates a robust **PySR** integration to evaluate trace numbers analytically back into exact mathematical fractions bridging numerical observations to algebraic structures directly.
- **`ramanujan/ai/dataset_generator.py`**: [NEW] Prepares vast caches of synthesized scaling relationships converting polynomial coefficient combinations toward pre-training array models identifying identity targets.
- **`ramanujan/ai/rl_agent.py`**: [NEW] Scaffolds the sequential validation loop utilizing target convergence precision limits scaling logarithmically to assert continuous mathematical rewards toward generative agents.

## 5. Comprehensive Edge Case Validation Testing (`tests/`)
Created a unified `<unittest>` testing suite asserting accuracy and scalability boundaries:
- **`test_gpu_enumerators.py`**: [NEW] Validates CPU parity against GPU parallel tensors. Injects absolute zero arrays (`[0,0,0]`) ensuring zero-division fractions smoothly bypass evaluations over mathematical infinity `NaN` errors. 
- **`test_asymptotic_filter.py`**: [NEW] Mathematically confirms Pringsheim/Worpitzky boundaries logic blocks divergence strictly.
- **`test_poly_domains.py`**: [NEW] Ascertains that gradient boundary scaling reliably shrinks stochastic continuous states effectively backward into constrained lattice integers.
- **`test_ai_modules.py`**: [NEW] Validations testing scalar generator arrays matching expected constraints.
- **`test_fr_expansion.py`**: [NEW] Tests PSLQ integer boundary expansion lengths safely tracking multi-dimensional matrix relations.
