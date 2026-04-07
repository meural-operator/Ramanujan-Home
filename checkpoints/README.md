# Pre-Trained Models & Checkpoints

This directory stores pre-trained Neural MCTS agents (`ActorCriticGCFNetwork`) used to guide the Ramanujan Machine's heuristic search.

## Active Checkpoints
*(No checkpoints currently available)*

## Deprecation Notice (April 2026)
Older `v1` checkpoints (e.g., `em_mcts.pt`, `em_mcts_test_latest.pt`) were trained using an RL environment that had a known warm-start bias in the `p_n` recurrence logic. These have been deleted as they do not navigate from a mathematically strict neutral starting state `p=0.0`. 

The `NeuralMCTSPolyDomain` now enforces a **schema v2** validation check. If you attempt to load a `v1` checkpoint, the pipeline will raise a `RuntimeError`.

## Retraining the Policy Network
To generate a valid `v2` checkpoint for the Euler-Mascheroni constant (or other targets), run the Neural MCTS training pipeline from the project root:

```powershell
python -m research_training.train_mcts_agent
```
Make sure training completes successfully before deploying the engine for GPU campaigns.
