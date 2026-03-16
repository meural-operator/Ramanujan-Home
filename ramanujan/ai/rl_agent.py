import math
import numpy as np

class GCFRewardEnvironment:
    """
    Defines the RL environment whose reward function is the numerical convergence rate 
    of the predicted GCF sequence to know mathematical constants.
    """
    def __init__(self, target_value: float, max_steps: int = 50):
        self.target_value = target_value
        self.max_steps = max_steps
        self.current_step = 0

    def calculate_convergence_reward(self, p_n: float, q_n: float) -> float:
        """
        Reward is strictly proportional to the number of digits matched with the target value.
        """
        if q_n == 0:
            return -10.0 # Heavy penalty for zero division

        predicted_val = p_n / q_n
        error = abs(self.target_value - predicted_val)
        
        if error == 0:
            return 100.0 # Perfect match
            
        # Logarithmic reward based on precision reached
        digits_accurate = -math.log10(error)
        
        # Reward is constrained to realistic digit bounds
        return max(0.0, min(100.0, digits_accurate))

    def step(self, action_a_n: float, action_b_n: float, prev_q, prev_p, q, p):
        """
        RL Step: Applies the chosen polynomial values for the current N depth.
        """
        next_q = action_a_n * q + action_b_n * prev_q
        next_p = action_a_n * p + action_b_n * prev_p
        
        reward = self.calculate_convergence_reward(next_p, next_q)
        done = self.current_step >= self.max_steps
        
        self.current_step += 1
        
        return (next_q, next_p), reward, done
