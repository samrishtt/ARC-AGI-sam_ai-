"""
ReAct (Reasoning and Acting) Agent for ARC-AGI-3.
This agent interacts with the `arc-agi` gym environments instead of generating static Python code.
"""
import sys
import os
import json

# Add parent directory to path so we can import our core LLM stack
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.llm import MultiProviderLLM
from arc_agi_3.memory import ReplayBuffer

class ArcAgi3Agent:
    def __init__(self, llm: MultiProviderLLM):
        self.llm = llm
        self.memory = ReplayBuffer()
        
    def _build_prompt(self, obs, time_steps_left: int) -> str:
        history = self.memory.get_structured_history()
        
        # In ARC-AGI-3, the observation contains the grid, but no rules.
        # We must prompt the LLM to analyze the grid and pick a specific action.
        sys_prompt = (
            "You are an active exploration agent deployed in an unknown physical grid puzzle.\n"
            "You must deduce the implicit goal of the environment by making moves and observing what happens.\n\n"
            "AVAILABLE ACTIONS:\n"
            "- move_up, move_down, move_left, move_right\n"
            "- select(x, y)\n"
            "- change_color(c)\n\n"
            "Format your response as valid JSON:\n"
            "{\n"
            "  'thoughts': 'Your physical hypothesis about the rules',\n"
            "  'action_type': 'move_up', \n"
            "  'action_args': {}\n"
            "}"
        )
        
        user_prompt = (
            f"{history}\n\n"
            f"--- CURRENT STATE ---\n"
            f"Observation Details: {obs}\n"
            f"Steps Remaining: {time_steps_left}\n\n"
            "What is your next action?"
        )
        return sys_prompt, user_prompt

    def play_episode(self, env, max_steps: int = 50):
        """Runs a single episode of interaction within the ARC-AGI-3 environment."""
        obs, info = env.reset()
        self.memory.start_episode(obs)
        self.llm.reset_provider()
        
        print("\n[ARC-AGI-3 Agent] Starting interactive episode...")
        
        for step in range(max_steps):
            sys_p, user_p = self._build_prompt(obs, max_steps - step)
            
            # Query LLM for the next physical action
            try:
                response = self.llm.generate(sys_p, user_p)
                
                # We would normally parse the JSON response here to extract action_type
                # For safety in this stub, we simulate an action parsing fallback
                action = self._parse_llm_action(response.content, env.action_space)
            except Exception as e:
                print(f"[Agent] LLM generation failed: {e}")
                break
                
            print(f"[Step {step}] Agent chose action: {action}")
            
            # Execute physical move in the environment
            try:
                next_obs, reward, done, truncated, info = env.step(action)
            except Exception as e:
                print(f"[Agent] Environment denied action: {e}")
                # We record the failure so the LLM learns the action is invalid next turn
                self.memory.record_step(action, obs, -1.0, False, {"error": str(e)})
                continue
                
            self.memory.record_step(action, next_obs, reward, done, info)
            obs = next_obs
            
            if done or truncated:
                print(f"[ARC-AGI-3 Agent] Episode finished! Final Reward: {reward}")
                break
                
        self.memory.end_episode()

    def _parse_llm_action(self, llm_output: str, action_space) -> Any:
        # Stub: Normally we regex extract the JSON block.
        # Here we just randomly sample the action space to avoid crashing if LLM outputs text.
        return action_space.sample()

