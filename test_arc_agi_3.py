"""Test script to load an ARC-AGI-3 environment and check compatibility with CSA."""
import arc_agi
import traceback

print("=" * 60)
print("  ARC-AGI-3 Interactive Environment Test")
print("=" * 60)

try:
    print("[1] Attempting to load an ARC-AGI-3 Environment...")
    
    # Initialize the ARC-AGI environment builder
    from arc_agi.envs.builder import make_env
    
    # Attempt to load a sample environment to see the structure
    env = make_env()
    obs, info = env.reset()
    
    print("\n[SUCCESS] Environment Loaded.")
    print(f"Observation Space : {env.observation_space}")
    print(f"Action Space      : {env.action_space}")
    print("\nInitial Observation:")
    print(obs)
    
except Exception as e:
    print(f"\n[ERROR] Could not load local environment: {e}")
    # traceback.print_exc()
    print("\nNOTE: ARC-AGI-3 is an API-based interactive RL environment, requiring an active app.agentops.ai project.")

print("\n--- CSA COMPATIBILITY CHECK ---")
print("1. CSA Input format   : Dict['train': [{'input': matrix, 'output': matrix}], 'test': ...]")
print("2. ARC-AGI-3 format   : Turn-based RL Environment (obs, reward, done, info = env.step(action))")
print("3. Structural Mismatch: CRITICAL")
print("   CSA writes complete `transform()` heuristic functions for static grids.")
print("   ARC-AGI-3 requires physical actions (move_x, select_tool, change_color) in real-time.")
print("=" * 60)
