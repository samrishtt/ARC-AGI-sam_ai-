"""
Episodic Memory Module for ARC-AGI-3 Interactive Environments.
Tracks the history of actions, observations, and rewards to allow the agent
to form hypotheses about the environment's hidden rules.
"""
from typing import List, Dict, Any

class ReplayBuffer:
    def __init__(self):
        self.episodes = []
        self.current_episode = []

    def start_episode(self, initial_observation: Any):
        self.current_episode = [{
            "step": 0,
            "action": None,
            "observation": initial_observation,
            "reward": 0.0,
            "done": False,
            "info": {}
        }]

    def record_step(self, action: Any, obs: Any, reward: float, done: bool, info: Dict):
        self.current_episode.append({
            "step": len(self.current_episode),
            "action": action,
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info
        })

    def end_episode(self):
        if self.current_episode:
            self.episodes.append(self.current_episode)
            self.current_episode = []

    def get_structured_history(self) -> str:
        """Serializes the current episode into a prompt-friendly string for the LLM."""
        if not self.current_episode:
            return "No history available."
            
        history_text = "--- EPISODIC MEMORY ---\n"
        for step in self.current_episode[-5:]: # Only show last 5 steps to avoid context limit
            act = step["action"] if step["action"] is not None else "START"
            reward = step["reward"]
            # We would format the observation grid here
            history_text += f"[Step {step['step']}] Action: {act} | Reward: {reward}\n"
            
        return history_text
