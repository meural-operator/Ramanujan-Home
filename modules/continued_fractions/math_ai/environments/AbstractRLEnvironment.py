from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict


class AbstractRLEnvironment(ABC):
    """
    Generalized abstract interface for Reinforcement Learning mathematical
    environments. Designed to emulate an OpenAI Gym `env` structure, making
    it plug-and-play for different discovery agents (DQN, PPO, Neural MCTS).
    """
    
    @abstractmethod
    def reset(self) -> Any:
        """
        Resets the environment state to begin a new episode/sequence generation.
        Returns the initial observation state.
        """
        pass
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        """
        Executes one time step within the environment based on the given action.
        
        Args:
            action: The action selected by the agent (e.g., coefficient values).
            
        Returns:
            observation (Any): The agent's observation of the current environment.
            reward (float): The amount of reward returned after previous action.
            done (bool): Whether the episode has ended (e.g., max depth reached).
            info (dict): Contains auxiliary diagnostic information.
        """
        pass
    
    @abstractmethod
    def calculate_reward(self, current_state: Any) -> float:
        """
        Specific mathematical logic defining the reward landscape.
        """
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Serialize the full internal state for MCTS node snapshots.
        Must capture all mutable state needed to resume from this exact point.
        Used by AlphaTensorMCTS to avoid O(depth) path replay.
        
        Returns:
            A dict containing all environment state variables.
        """
        pass

    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restore environment to a previously captured snapshot.
        After set_state(s), the environment must behave identically
        to when get_state() returned s.
        
        Args:
            state: A dict previously returned by get_state().
        """
        pass
