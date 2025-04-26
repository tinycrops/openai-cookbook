#!/usr/bin/env python3
"""
Ablation study to compare different agentic_insight configurations
This script compares:
1. Original table-based model with greedy selection
2. Enhanced neural world model 
3. DQN action selection
4. Full integration with Spotify

Run this script to generate performance comparison graphs
"""

import os
import sys
import asyncio
import random
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from parent directory
from agentic_insight_pipeline import (
    EnhancedWorldModel, DQNAgent, ExperimentTracker,
    calculate_composite_reward, init_db, signal_recorder
)

class MockAction:
    """Simulated actions for testing"""
    
    ACTION_TYPES = [
        "suggestion", "question", "exercise", 
        "reminder", "focus_music", "deep_breathing"
    ]
    
    TEMPLATES = {
        "suggestion": [
            "Consider taking a short break to refresh your mind.",
            "Try the Pomodoro technique: 25 minutes of focus followed by a 5-minute break.",
            "Make a to-do list to better organize your tasks.",
            "Consider reducing distractions in your environment."
        ],
        "question": [
            "Are you feeling focused on your current task?",
            "What's your most important goal for today?",
            "Are you finding it difficult to concentrate?",
            "Have you been taking regular breaks?"
        ],
        "exercise": [
            "Try a quick desk stretching routine for 2 minutes.",
            "Stand up and do 10 jumping jacks to boost circulation.",
            "Practice 5 deep breaths with your eyes closed.",
            "Roll your shoulders backward and forward 10 times."
        ],
        "reminder": [
            "Don't forget to drink water regularly.",
            "Remember to check your posture while sitting.",
            "It's time to refocus on your primary task.",
            "Consider switching tasks if you've been stuck for too long."
        ],
        "focus_music": [
            "I recommend playing some focus music to enhance concentration.",
            "Studies show that instrumental music can boost productivity.",
            "Would you like me to play a concentration-enhancing playlist?",
            "Listening to focus music may help you maintain attention."
        ],
        "deep_breathing": [
            "Take a moment for a quick breathing exercise.",
            "Try box breathing: inhale 4 counts, hold 4, exhale 4, hold 4.",
            "A 1-minute mindfulness exercise can help clear your mind.",
            "Practice diaphragmatic breathing to reduce stress and improve focus."
        ]
    }
    
    @classmethod
    def generate_random(cls, count=3):
        """Generate random actions"""
        actions = []
        for _ in range(count):
            action_type = random.choice(cls.ACTION_TYPES)
            content = random.choice(cls.TEMPLATES[action_type])
            predicted_effect = "Expected to improve focus and productivity."
            
            actions.append({
                "type": action_type,
                "content": content,
                "predicted_effect": predicted_effect
            })
        return actions

class MockAgent:
    """Simulated agent for ablation studies"""
    
    def __init__(self, user_id="test_user", config=None):
        self.user_id = user_id
        self.config = config or {
            "use_neural_model": True,
            "use_dqn": True,
            "use_spotify": False,
            "cycles": 100
        }
        
        # Components
        self.world_model = None
        self.dqn_agent = None
        self.experiment_tracker = None
        
        # State
        self.state = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "cycles": 0,
            "total_reward": 0
        }
        
        # Initialize components based on config
        self._init_components()
    
    def _init_components(self):
        """Initialize components based on configuration"""
        # Always initialize DB first
        init_db()
        
        # Choose world model
        if self.config["use_neural_model"]:
            self.world_model = EnhancedWorldModel(self.user_id)
        else:
            # Simple world model (just the table part)
            self.world_model = EnhancedWorldModel(self.user_id)
            # Disable neural components
            self.world_model.nn_model = None
        
        # Choose action selector
        if self.config["use_dqn"]:
            self.dqn_agent = DQNAgent(self.user_id)
        
        # Setup experiment tracker
        self.experiment_tracker = ExperimentTracker(self.user_id)
        self.experiment_tracker.start_experiment(
            f"ablation_{int(time.time())}", 
            self.config
        )
    
    def _select_action(self, state_emb, actions):
        """Select action based on configuration"""
        if self.config["use_dqn"] and self.dqn_agent:
            # Use DQN for selection
            selected_idx = self.dqn_agent.select_action(state_emb, actions, self.world_model)
        else:
            # Use greedy world model selection
            best_idx = 0
            best_reward = float('-inf')
            
            for i, action in enumerate(actions):
                action_json = json.dumps(action)
                _, reward = self.world_model.predict_outcome(
                    json.dumps(self.state), action_json
                )
                
                if reward > best_reward:
                    best_reward = reward
                    best_idx = i
            
            # If model has no good prediction, use random
            if best_reward == float('-inf'):
                selected_idx = random.randint(0, len(actions) - 1)
            else:
                selected_idx = best_idx
        
        return selected_idx
    
    def _calculate_reward(self, action):
        """Calculate simulated reward for an action"""
        # Base reward component
        base_reward = random.uniform(-0.3, 0.3)
        
        # Action type component
        action_type_rewards = {
            "suggestion": 0.1,
            "question": 0.0,
            "exercise": 0.2,
            "reminder": 0.05,
            "focus_music": 0.3 if self.config["use_spotify"] else 0.1,
            "deep_breathing": 0.15
        }
        
        action_reward = action_type_rewards.get(action["type"], 0.0)
        
        # Previous action influence (penalize repetition)
        repetition_penalty = 0.0
        if self.state["actions"] and self.state["actions"][-1]["type"] == action["type"]:
            repetition_penalty = -0.2
        
        # Learning component (better rewards over time)
        learning_bonus = min(0.3, self.state["cycles"] / (self.config["cycles"] * 2))
        
        # Random noise
        noise = random.uniform(-0.1, 0.1)
        
        # Calculate final reward
        reward = base_reward + action_reward + repetition_penalty + learning_bonus + noise
        
        # Constrain to [-1, 1] range
        return max(-1.0, min(1.0, reward))
    
    def run_cycle(self):
        """Run a single cycle of the agent"""
        # Update state
        self.state["cycles"] += 1
        
        # Generate mock state embedding
        state_emb = np.random.randn(768).astype(np.float32)
        
        # Generate possible actions
        actions = MockAction.generate_random(count=3)
        
        # Select action
        selected_idx = self._select_action(state_emb, actions)
        selected_action = actions[selected_idx]
        
        # Calculate reward
        reward = self._calculate_reward(selected_action)
        
        # Update state
        self.state["actions"].append(selected_action)
        self.state["rewards"].append(reward)
        self.state["total_reward"] += reward
        
        # Calculate action embedding
        action_json = json.dumps(selected_action)
        action_emb = self.world_model._get_action_embedding(action_json)
        
        # Update world model
        self.world_model.update(
            json.dumps(self.state),
            action_json,
            json.dumps(self.state),  # Updated state
            reward
        )
        
        # Update DQN if enabled
        if self.config["use_dqn"] and self.dqn_agent:
            next_state_emb = np.random.randn(768).astype(np.float32)  # Mock next state
            self.dqn_agent.update(state_emb, action_emb, reward, next_state_emb)
        
        # Log to experiment tracker
        self.experiment_tracker.log_reward(reward, selected_action["type"], reward)
        
        # Log metrics
        signal_recorder.record_signal(
            "ablation_reward", 
            reward,
            {
                "action_type": selected_action["type"],
                "cycle": self.state["cycles"],
                "use_neural_model": self.config["use_neural_model"],
                "use_dqn": self.config["use_dqn"],
                "use_spotify": self.config["use_spotify"]
            }
        )
        
        return {
            "action": selected_action,
            "reward": reward,
            "cycle": self.state["cycles"]
        }
    
    def run_experiment(self):
        """Run full experiment"""
        results = []
        
        try:
            for _ in range(self.config["cycles"]):
                result = self.run_cycle()
                results.append(result)
                
                # Print progress every 10 cycles
                if result["cycle"] % 10 == 0:
                    avg_reward = sum(self.state["rewards"][-10:]) / 10
                    print(f"Cycle {result['cycle']}/{self.config['cycles']} - Avg reward: {avg_reward:.3f}")
            
            # End experiment and return results
            final_results = self.experiment_tracker.end_experiment()
            return {
                "config": self.config,
                "results": results,
                "summary": final_results,
                "total_reward": self.state["total_reward"],
                "avg_reward": self.state["total_reward"] / self.config["cycles"]
            }
            
        except Exception as e:
            print(f"Error running experiment: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to end experiment
            if self.experiment_tracker.current_experiment:
                self.experiment_tracker.end_experiment()
            
            return {
                "config": self.config,
                "error": str(e),
                "results": results,
                "total_reward": self.state["total_reward"],
                "avg_reward": self.state["total_reward"] / max(1, self.state["cycles"])
            }

def run_ablation_study(num_cycles=100):
    """Run full ablation study with all configurations"""
    configurations = [
        # Baseline (simple table model with greedy selection)
        {
            "name": "Baseline",
            "use_neural_model": False,
            "use_dqn": False,
            "use_spotify": False,
            "cycles": num_cycles
        },
        # Neural world model only
        {
            "name": "Neural World Model",
            "use_neural_model": True,
            "use_dqn": False,
            "use_spotify": False,
            "cycles": num_cycles
        },
        # DQN action selection only
        {
            "name": "DQN Action Selection",
            "use_neural_model": False,
            "use_dqn": True,
            "use_spotify": False,
            "cycles": num_cycles
        },
        # Full system (neural model + DQN + Spotify)
        {
            "name": "Full System",
            "use_neural_model": True,
            "use_dqn": True,
            "use_spotify": True,
            "cycles": num_cycles
        }
    ]
    
    all_results = {}
    action_type_rewards = defaultdict(lambda: defaultdict(list))
    
    # Run each configuration
    for config in configurations:
        print(f"\n=== Running configuration: {config['name']} ===\n")
        agent = MockAgent(user_id=f"ablation_{int(time.time())}", config=config)
        results = agent.run_experiment()
        all_results[config["name"]] = results
        
        # Collect action type rewards for analysis
        for i, result in enumerate(results["results"]):
            action_type = result["action"]["type"]
            reward = result["reward"]
            action_type_rewards[config["name"]][action_type].append((i, reward))
    
    # Create visualization
    create_ablation_plots(all_results, action_type_rewards, num_cycles)
    
    return all_results

def create_ablation_plots(all_results, action_type_rewards, num_cycles):
    """Create visualization of ablation study results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot 1: Cumulative rewards
    plt.figure(figsize=(12, 8))
    
    for config_name, results in all_results.items():
        rewards = [r["reward"] for r in results["results"]]
        cumulative = np.cumsum(rewards)
        plt.plot(range(1, len(cumulative) + 1), cumulative, label=f"{config_name} (Total: {cumulative[-1]:.2f})")
    
    plt.title("Cumulative Rewards Across Configurations")
    plt.xlabel("Interaction Cycle")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"ablation_cumulative_rewards_{timestamp}.png")
    
    # Plot 2: Moving average rewards
    plt.figure(figsize=(12, 8))
    window_size = 10
    
    for config_name, results in all_results.items():
        rewards = [r["reward"] for r in results["results"]]
        moving_avg = [sum(rewards[max(0, i-window_size):i]) / min(i, window_size) for i in range(1, len(rewards) + 1)]
        plt.plot(range(1, len(moving_avg) + 1), moving_avg, label=config_name)
    
    plt.title(f"Moving Average Rewards (Window Size: {window_size})")
    plt.xlabel("Interaction Cycle")
    plt.ylabel(f"Average Reward (Last {window_size} Cycles)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"ablation_moving_avg_rewards_{timestamp}.png")
    
    # Plot 3: Action type distribution
    plt.figure(figsize=(12, 8))
    
    action_counts = {}
    for config_name, results in all_results.items():
        action_counts[config_name] = defaultdict(int)
        for result in results["results"]:
            action_type = result["action"]["type"]
            action_counts[config_name][action_type] += 1
    
    # Get all unique action types
    all_action_types = set()
    for counts in action_counts.values():
        all_action_types.update(counts.keys())
    
    x = range(len(all_action_types))
    width = 0.8 / len(action_counts)
    action_types_list = sorted(list(all_action_types))
    
    for i, (config_name, counts) in enumerate(action_counts.items()):
        values = [counts.get(action_type, 0) for action_type in action_types_list]
        plt.bar([p + i * width for p in x], values, width, label=config_name)
    
    plt.title("Action Type Distribution by Configuration")
    plt.xlabel("Action Type")
    plt.ylabel("Count")
    plt.xticks([p + width * (len(action_counts) - 1) / 2 for p in x], action_types_list)
    plt.legend()
    plt.savefig(f"ablation_action_distribution_{timestamp}.png")
    
    # Plot 4: Action type effectiveness (reward per action type)
    plt.figure(figsize=(12, 8))
    
    for config_name, action_data in action_type_rewards.items():
        for action_type, rewards in action_data.items():
            # Only plot if we have enough data points
            if len(rewards) >= 5:
                cycles, reward_values = zip(*rewards)
                plt.scatter(cycles, reward_values, alpha=0.5, label=f"{config_name} - {action_type}")
    
    plt.title("Action Type Effectiveness by Configuration")
    plt.xlabel("Cycle")
    plt.ylabel("Reward")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"ablation_action_effectiveness_{timestamp}.png")
    
    print(f"Plots saved with timestamp: {timestamp}")

async def main():
    """Main entry point"""
    print("\nRunning Agentic Insights Ablation Study\n")
    
    num_cycles = 100
    print(f"Running {num_cycles} cycles for each configuration...")
    
    results = run_ablation_study(num_cycles)
    
    # Print summary
    print("\n=== Summary ===")
    for config_name, result in results.items():
        print(f"{config_name}: Avg Reward = {result['avg_reward']:.3f}, Total = {result['total_reward']:.3f}")
    
    # Clean up
    signal_recorder.close()
    
    print("\nAblation study complete. Check the generated plots for detailed results.")

if __name__ == "__main__":
    asyncio.run(main()) 