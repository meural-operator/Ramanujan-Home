"""
Euler-Mascheroni Specific RL Environment.

The Euler-Mascheroni constant γ ≈ 0.5772156649015328...
is one of the deepest unsolved constants in mathematics.
No simple GCF formula with polynomial coefficients has been proven to converge to γ.

This environment shapes the reward to:
  1. Reward digit-accuracy (primary signal)
  2. Bonus for *improving* convergence rate step-over-step (shaped exploration reward)
  3. Numerical overflow guard (early termination with penalty)
  4. Normalized state observation for stable neural network input (inherited from base)
"""
import math
import mpmath
import numpy as np
from typing import Tuple, Dict, Any

from .GCFRewardEnvironment import GCFRewardEnvironment

# High-precision Euler-Mascheroni constant (mpmath's reference value)
_GAMMA = float(mpmath.euler)  # 0.5772156649015328606...


class EulerMascheroniEnvironment(GCFRewardEnvironment):
    """
    Reinforcement learning environment for discovering GCF representations of γ.
    
    State:
        Inherited from GCFRewardEnvironment._get_obs():
        [sign(prev_q)*log1p(|prev_q|), sign(prev_p)*log1p(|prev_p|),
         sign(q)*log1p(|q|),           sign(p)*log1p(|p|)]
        Log-scaling prevents float overflow while preserving magnitude ordering.
    
    Action:
        Continuous 2D vector (a_n_proxy, b_n_proxy) representing the polynomial
        coefficient scaling factors for the current GCF depth.
    
    Reward:
        - Primary: digits of precision gained (log10 scale) vs target γ
        - Bonus: +2 for each digit of improvement over the previous best precision
        - Penalty: -20 for numerical overflow (|p| or |q| > 1e12 before normalization)
        - Penalty: -10 for q = 0 division
    """

    env_name: str = "euler_mascheroni"
    target_val: float = _GAMMA

    def __init__(self, max_steps: int = 100):
        # Initialize base with γ as target — p=0.0 is the standard GCF initial condition
        super().__init__(target_value=_GAMMA, max_steps=max_steps)
        self.best_digits = 0.0

    def reset(self) -> np.ndarray:
        obs = super().reset()
        # Standard GCF initial condition: p=0.0, q=1.0
        # The agent must navigate the recurrence to converge toward γ from neutral start.
        # NOTE: Previous versions used self.p = _GAMMA (warm-start) which gave the agent
        # a free maximum-precision signal at step 0, inflating reported rewards.
        # Checkpoints trained with warm-start (schema v1) are incompatible.
        self.best_digits = 0.0
        return obs

    def calculate_reward(self, p: float, q: float) -> float:
        """Primary reward: log10(1 / |γ - p/q|), i.e. digits of precision."""
        if abs(q) < 1e-15:
            return -10.0

        predicted = p / q
        error = abs(self.target_value - predicted)

        if error == 0.0:
            return 100.0

        digits = -math.log10(error + 1e-300)
        return max(0.0, min(100.0, digits))

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Apply one GCF recurrence step using the action as (a_n, b_n) proxies.
        
        The action is treated as a *scaling multiplier* applied to the running
        convergent before update — this lets the agent learn to guide the GCF
        trajectory direction without prescribing exact integer coefficients.
        """
        a_n_proxy, b_n_proxy = float(action[0]), float(action[1])

        # Overflow guard before update
        if abs(self.q) > 1e12 or abs(self.p) > 1e12:
            return self._get_obs(), -20.0, True, {"overflow": True}

        # GCF recurrence: q_n = a_n * q_{n-1} + b_n * q_{n-2}
        next_q = a_n_proxy * self.q + b_n_proxy * self.prev_q
        next_p = a_n_proxy * self.p + b_n_proxy * self.prev_p

        # Periodic scaling to keep magnitudes tractable
        scale = max(abs(next_q), 1.0)
        next_q /= scale
        next_p /= scale
        self.prev_q = self.q / scale
        self.prev_p = self.p / scale
        self.q = next_q
        self.p = next_p

        # Compute primary reward
        primary_reward = self.calculate_reward(self.p, self.q)

        # Convergence-rate bonus: extra reward for improving our best digit count
        improvement_bonus = 0.0
        if primary_reward > self.best_digits:
            improvement_bonus = 2.0 * (primary_reward - self.best_digits)
            self.best_digits = primary_reward

        reward = primary_reward + improvement_bonus

        self.current_step += 1
        done = self.current_step >= self.max_steps

        return self._get_obs(), reward, done, {
            "digits_accurate": primary_reward,
            "improvement_bonus": improvement_bonus,
        }

    def get_state(self) -> Dict[str, Any]:
        """Serialize full state including best_digits for MCTS snapshots."""
        state = super().get_state()
        state['best_digits'] = self.best_digits
        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore state including best_digits."""
        super().set_state(state)
        self.best_digits = state.get('best_digits', 0.0)
