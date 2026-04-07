import os
import logging
import torch
from modules.continued_fractions.domains.CartesianProductPolyDomain import CartesianProductPolyDomain
from modules.continued_fractions.math_ai.models.actor_critic import ActorCriticGCFNetwork
from modules.continued_fractions.math_ai.environments.GCFRewardEnvironment import GCFRewardEnvironment
from modules.continued_fractions.math_ai.agents.alpha_tensor_mcts import AlphaTensorMCTS

logger = logging.getLogger(__name__)

# Default checkpoint path (relative to this file's directory)
_DEFAULT_CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'checkpoints')

class NeuralMCTSPolyDomain(CartesianProductPolyDomain):
    """
    AlphaGo/DeepMind-style Neural MCTS PolyDomain for Ramanujan Machine.
    Uses an Actor-Critic Policy Network to simulate trajectory convergence
    and bounds the brute-force GPU exhaustion to the most mathematically 
    promising domains (Upper Confidence Bound selections).
    
    Integration flow:
        1. Initialize GCFRewardEnvironment for the target constant
        2. Load pre-trained ActorCriticGCFNetwork checkpoint
        3. Run AlphaTensorMCTS.get_action_for_bounds() to get tightened coefficient bounds
        4. Pass tightened bounds to the GPU enumerator
    """
    def __init__(self, a_deg, a_coef_range, b_deg, b_coef_range, target_val, 
                 mcts_simulations=500, checkpoint_path=None, *args, **kwargs):
        self.target_val = target_val
        self.mcts_simulations = mcts_simulations
        self.checkpoint_path = checkpoint_path
        
        # We start with the massive Cartesian bounds defined by the user
        super().__init__(a_deg, a_coef_range, b_deg, b_coef_range, *args, **kwargs)
        
    def _setup_metadata(self):
        self._run_neural_mcts_optimization()
        super()._setup_metadata()

    def _find_checkpoint(self):
        """Locate a valid checkpoint file, respecting version metadata."""
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            return self.checkpoint_path
        
        # Search default checkpoint directory
        default_path = os.path.join(_DEFAULT_CHECKPOINT_DIR, 'em_mcts.pt')
        if os.path.exists(default_path):
            return default_path
        
        return None

    def _load_network_with_validation(self, network, checkpoint_path):
        """
        Load checkpoint with version validation.
        Raises RuntimeError if checkpoint was trained with incompatible settings.
        """
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Version-guard: reject checkpoints trained with warm-start bias (v1)
        meta = state.get('_checkpoint_meta', {})
        schema_version = meta.get('schema_version', 1)
        if schema_version < 2:
            raise RuntimeError(
                f"Checkpoint '{checkpoint_path}' was trained with schema v{schema_version} "
                f"(warm-start bias). It is incompatible with the current environment. "
                f"Please retrain with: python -m research_training.train_mcts_agent"
            )
        
        # Load the model weights from the checkpoint
        if 'model_state_dict' in state:
            network.load_state_dict(state['model_state_dict'])
        else:
            # Legacy format: entire state dict is the model weights
            network.load_state_dict(state)
        
        logger.info(f"Loaded Neural MCTS checkpoint from: {checkpoint_path} (schema v{schema_version})")
        return True

    def _run_neural_mcts_optimization(self):
        """
        Runs the full Deep Reinforcement Learning pipeline to shrink the bounds.
        Uses AlphaTensorMCTS with a trained ActorCriticGCFNetwork to intelligently
        narrow the coefficient search space before GPU exhaustive enumeration.
        """
        print(f"Initializing Neural-Guided MCTS Search (Simulations: {self.mcts_simulations})...")
        
        # 1. Init RL Environment for the given mathematical constant
        env = GCFRewardEnvironment(target_value=self.target_val, max_steps=100)
        
        # 2. Init the Policy-Value AI Network 
        network = ActorCriticGCFNetwork(state_dim=4, hidden_dim=256, action_dim=2)
        
        # 3. Load pre-trained weights (critical for non-random search behavior)
        checkpoint_path = self._find_checkpoint()
        network_loaded = False
        if checkpoint_path:
            try:
                network_loaded = self._load_network_with_validation(network, checkpoint_path)
            except RuntimeError as e:
                logger.warning(f"Checkpoint rejected: {e}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint '{checkpoint_path}': {e}")
        
        if not network_loaded:
            logger.warning(
                "No valid pre-trained checkpoint found. Neural MCTS will run with "
                "untrained weights (structured random exploration). For production use, "
                "train a checkpoint first."
            )
        
        # 4. Init AlphaTensor MCTS Agent
        mcts_agent = AlphaTensorMCTS(env=env, network=network, num_simulations=self.mcts_simulations)
        
        # 5. Use the high-level bounds API — handles search(), action→bounds conversion,
        #    and radius computation internally
        initial_state = env.reset()
        
        try:
            new_a_range, new_b_range = mcts_agent.get_action_for_bounds(
                initial_state=initial_state,
                original_a_range=self.a_coef_range,
                original_b_range=self.b_coef_range,
                radius_multiplier=3.0
            )
        except Exception as e:
            logger.warning(f"Neural MCTS search failed: {e}. Falling back to full Cartesian bounds.")
            return
        
        # 6. Apply the narrowed bounds
        self.a_coef_range = new_a_range
        self.b_coef_range = new_b_range
        
        print(f"Neural Search Complete. Narrowed bounds:")
        print(f"  a_coef_range: {self.a_coef_range}")
        print(f"  b_coef_range: {self.b_coef_range}")
