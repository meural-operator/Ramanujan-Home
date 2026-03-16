import unittest
import os
import json
import numpy as np
from ramanujan.ai.dataset_generator import MathematicalDatasetGenerator
from ramanujan.ai.rl_agent import GCFRewardEnvironment

class TestAIModules(unittest.TestCase):
    def test_dataset_generator(self):
        gen = MathematicalDatasetGenerator(num_samples=5)
        path = "test_data.json"
        gen.build_synthetic_dataset(path)
        
        self.assertTrue(os.path.exists(path))
        with open(path, 'r') as f:
            data = json.load(f)
            
        self.assertEqual(len(data), 5)
        self.assertIn("a_coefs", data[0])
        self.assertIn("b_coefs", data[0])
        
        os.remove(path)
        
    def test_rl_reward_environment(self):
        # Target Pi
        env = GCFRewardEnvironment(target_value=3.1415926535, max_steps=10)
        
        # Exact match
        reward1 = env.calculate_convergence_reward(31415926535, 10000000000)
        self.assertEqual(reward1, 100.0)
        
        # Distant match
        reward2 = env.calculate_convergence_reward(3, 1)
        self.assertTrue(reward2 > 0 and reward2 < 2.0)
        
        # Zero division penalty
        reward3 = env.calculate_convergence_reward(1, 0)
        self.assertEqual(reward3, -10.0)
        
    def test_rl_step(self):
        env = GCFRewardEnvironment(target_value=3.0, max_steps=2)
        
        # Step 1
        (q, p), reward, done = env.step(action_a_n=1, action_b_n=0, prev_q=0, prev_p=1, q=1, p=2)
        self.assertEqual(q, 1)
        self.assertEqual(p, 2)
        self.assertFalse(done)
        
        # Step 2
        (q, p), reward, done = env.step(action_a_n=1, action_b_n=0, prev_q=1, prev_p=2, q=1, p=2)
        self.assertTrue(done)

if __name__ == '__main__':
    unittest.main()
