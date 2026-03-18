import gymnasium as gym
import numpy as np
from typing import Optional, Sequence, Dict, Any, Tuple
from semantic.source.semantic_state_provider import SemanticStateProvider

class SemanticGraphWrappedEnv(gym.Env):
    """
    Wraps a base Gym/Sinergym environment and augments its observation
    with a semantic latent vector z_t produced by SemanticStateProvider.

    Modes:
        - "concat": observation = [base_obs ; z_t]
        - "latent": observation = z_t
    """

    def __init__(self,
                 base_env: gym.Env,
                 state_provider: SemanticStateProvider,
                 mode: str = "concat",
                 obs_var_names: Optional[Sequence[str]] = None):
        """
        Initialize the wrapper.

        Args:
            base_env (gym.Env): The base environment to wrap.
            state_provider (SemanticStateProvider): Initialized state provider.
            mode (str): "concat" or "latent".
            obs_var_names (list[str], optional): List of variable names for the base observation.
        """
        self.base_env = base_env
        self.state_provider = state_provider
        self.mode = mode
        
        # Determine observation variable names
        if obs_var_names is None:
            self.obs_var_names = getattr(base_env, "observation_variables", None)
            if self.obs_var_names is None:
                # Try to infer from wrapper if possible, or raise error
                # Some Sinergym envs might have it in a different place
                if hasattr(base_env, "unwrapped"):
                    self.obs_var_names = getattr(base_env.unwrapped, "observation_variables", None)
                
                if self.obs_var_names is None:
                     raise ValueError("Could not determine observation variable names from base_env. Please provide obs_var_names.")
        else:
            self.obs_var_names = obs_var_names

        # Update observation space
        self.latent_dim = state_provider.latent_dim
        
        # Get base observation space bounds
        # We assume Box space for now
        if not isinstance(base_env.observation_space, gym.spaces.Box):
             raise ValueError("Base environment observation space must be gym.spaces.Box")

        if self.mode == "concat":
            low = np.concatenate(
                [base_env.observation_space.low,
                 np.full(self.latent_dim, -np.inf, dtype=np.float32)]
            )
            high = np.concatenate(
                [base_env.observation_space.high,
                 np.full(self.latent_dim, np.inf, dtype=np.float32)]
            )
            self.observation_space = gym.spaces.Box(
                low=low, high=high, dtype=np.float32
            )
        elif self.mode == "latent":
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.latent_dim,),
                dtype=np.float32
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.action_space = base_env.action_space

    def _build_snapshot(self, obs: np.ndarray) -> Dict[str, float]:
        """
        Convert observation array to snapshot dictionary.
        """
        return {name: float(obs[i]) for i, name in enumerate(self.obs_var_names)}

    def _combine(self, base_obs: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Combine base observation and latent vector based on mode.
        """
        if self.mode == "concat":
            return np.concatenate([base_obs, z])
        elif self.mode == "latent":
            return z
        return base_obs

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.
        """
        base_obs, info = self.base_env.reset(**kwargs)
        
        snapshot = self._build_snapshot(base_obs)
        z = self.state_provider.get_state_from_snapshot(snapshot)
        
        obs_aug = self._combine(base_obs, z)
        info["semantic_state"] = z
        
        return obs_aug, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Step the environment.
        """
        base_obs, reward, terminated, truncated, info = self.base_env.step(action)
        
        snapshot = self._build_snapshot(base_obs)
        z = self.state_provider.get_state_from_snapshot(snapshot)
        
        obs_aug = self._combine(base_obs, z)
        info["semantic_state"] = z
        
        return obs_aug, reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        return self.base_env.render(*args, **kwargs)

    def close(self):
        return self.base_env.close()
