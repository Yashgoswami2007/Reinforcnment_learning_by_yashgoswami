import bpy
import random
import math
import numpy as np
import csv
import os
from datetime import datetime

# --- PPO and Experience Replay Settings ---
EXPERIENCE_BUFFER_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.99
LAMBDA = 0.95
PPO_CLIP = 0.2
LEARNING_RATE = 0.0003
PPO_EPOCHS = 4

# --- Car Control Settings ---
DRIVE_VALUE = 5.0
STEERING_VALUE = 15.0
MAX_STEPS = 1000

# --- State and Training ---
class PPOCarAgent:
    def __init__(self, state_size=6, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        self.experience_buffer = []
        self.episode_data = []
        
        # Simple neural network weights (in practice, you'd use a proper NN library)
        self.policy_weights = np.random.randn(state_size, action_size) * 0.1
        self.value_weights = np.random.randn(state_size, 1) * 0.1
        
        # PPO state
        self.old_policy_weights = self.policy_weights.copy()
        
    def get_action(self, state):
        """Sample action from policy"""
        logits = np.dot(state, self.policy_weights)
        probs = self.softmax(logits)
        action = np.random.choice(self.action_size, p=probs)
        return action, probs[action]
    
    def get_value(self, state):
        """Get state value"""
        return np.dot(state, self.value_weights)[0]
    
    def softmax(self, x):
        """Softmax function"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def store_experience(self, state, action, reward, next_state, done, prob):
        """Store experience in buffer"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'prob': prob
        }
        self.experience_buffer.append(experience)
        
        # Trim buffer if too large
        if len(self.experience_buffer) > EXPERIENCE_BUFFER_SIZE:
            self.experience_buffer.pop(0)
    
    def save_to_csv(self):
        """Save experience buffer to CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"car_ppo_experience_{timestamp}.csv"
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['episode', 'step', 'state', 'action', 'reward', 'next_state', 'done', 'prob'])
            
            for i, exp in enumerate(self.experience_buffer):
                writer.writerow([
                    i, i,  # episode, step
                    ','.join(map(str, exp['state'])),
                    exp['action'],
                    exp['reward'],
                    ','.join(map(str, exp['next_state'])),
                    exp['done'],
                    exp['prob']
                ])
        
        print(f"💾 Experience saved to {filename}")
    
    def compute_advantages(self, rewards, values, next_value, dones):
        """Compute advantages using GAE"""
        advantages = np.zeros(len(rewards))
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_value
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + GAMMA * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + GAMMA * LAMBDA * last_advantage * (1 - dones[t])
            last_advantage = advantages[t]
        
        return advantages
    
    def update_policy(self):
        """PPO policy update"""
        if len(self.experience_buffer) < BATCH_SIZE:
            return
        
        # Convert experiences to arrays
        states = np.array([exp['state'] for exp in self.experience_buffer])
        actions = np.array([exp['action'] for exp in self.experience_buffer])
        rewards = np.array([exp['reward'] for exp in self.experience_buffer])
        next_states = np.array([exp['next_state'] for exp in self.experience_buffer])
        dones = np.array([exp['done'] for exp in self.experience_buffer])
        old_probs = np.array([exp['prob'] for exp in self.experience_buffer])
        
        # Compute advantages
        values = np.array([self.get_value(state) for state in states])
        next_value = self.get_value(next_states[-1]) if not dones[-1] else 0
        advantages = self.compute_advantages(rewards, values, next_value, dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(PPO_EPOCHS):
            # Sample batch
            indices = np.random.choice(len(self.experience_buffer), BATCH_SIZE)
            
            batch_states = states[indices]
            batch_actions = actions[indices]
            batch_advantages = advantages[indices]
            batch_old_probs = old_probs[indices]
            
            # Compute new probabilities
            for state, action, advantage, old_prob in zip(batch_states, batch_actions, batch_advantages, batch_old_probs):
                logits = np.dot(state, self.policy_weights)
                new_probs = self.softmax(logits)
                new_prob = new_probs[action]
                
                # PPO ratio
                ratio = new_prob / (old_prob + 1e-8)
                
                # PPO loss
                surr1 = ratio * advantage
                surr2 = np.clip(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * advantage
                policy_loss = -np.minimum(surr1, surr2)
                
                # Update weights (simplified gradient update)
                self.policy_weights -= LEARNING_RATE * policy_loss * state.reshape(-1, 1) * action
                
                # Value function update
                target = advantage + self.get_value(state)
                value_loss = (target - self.get_value(state)) ** 2
                self.value_weights -= LEARNING_RATE * value_loss * state.reshape(-1, 1)
        
        print("🔄 PPO Policy Updated!")

# --- Car Environment ---
class CarEnvironment:
    def __init__(self):
        self.agent = PPOCarAgent()
        self.current_episode = 0
        self.current_step = 0
        self.current_state = None
        self.total_reward = 0
        self.training_active = True
        
    def get_state(self):
        """Get current state of the car"""
        try:
            rig = bpy.context.scene.sna_rbc_rig_collection[0]
            
            # Simple state representation: [drive, steering, velocity_x, velocity_y, position_x, position_z]
            state = [
                rig.rig_drivers.drive / DRIVE_VALUE,  # Normalized drive
                rig.rig_drivers.steering / STEERING_VALUE,  # Normalized steering
                0.0,  # Would be actual velocity components
                0.0,
                0.0,  # Would be actual position
                0.0
            ]
            
            return np.array(state)
        except:
            return np.zeros(6)
    
    def calculate_reward(self, state, action):
        """Calculate reward for current state and action"""
        reward = 0.0
        
        # Reward for moving forward
        if state[0] > 0:  # Positive drive
            reward += 0.1
        
        # Penalize excessive steering
        reward -= 0.05 * abs(state[1])
        
        # Reward for smooth operation (small actions)
        if action == 1:  # Neutral action
            reward += 0.05
        
        # Small penalty per step to encourage efficiency
        reward -= 0.01
        
        return reward
    
    def map_action_to_control(self, action):
        """Map discrete action to car controls"""
        if action == 0:  # Forward
            return DRIVE_VALUE * 0.8, 0.0
        elif action == 1:  # Forward with right turn
            return DRIVE_VALUE * 0.6, STEERING_VALUE * 0.7
        elif action == 2:  # Forward with left turn
            return DRIVE_VALUE * 0.6, -STEERING_VALUE * 0.7
    
    def reset_episode(self):
        """Reset episode data"""
        self.current_step = 0
        self.total_reward = 0
        self.current_state = self.get_state()
        self.current_episode += 1
        
    def step(self):
        """Execute one training step"""
        if not self.training_active:
            return
        
        # Get action from policy
        action, action_prob = self.agent.get_action(self.current_state)
        
        # Apply action to car
        drive, steering = self.map_action_to_control(action)
        try:
            rig = bpy.context.scene.sna_rbc_rig_collection[0]
            rig.rig_drivers.drive = drive
            rig.rig_drivers.steering = steering
        except Exception as e:
            print(f"Car control error: {e}")
        
        # Get new state
        next_state = self.get_state()
        
        # Calculate reward
        reward = self.calculate_reward(next_state, action)
        self.total_reward += reward
        
        # Check if episode is done
        done = self.current_step >= MAX_STEPS
        
        # Store experience
        self.agent.store_experience(
            self.current_state, action, reward, next_state, done, action_prob
        )
        
        # Update state
        self.current_state = next_state
        self.current_step += 1
        
        # Print progress
        if self.current_step % 50 == 0:
            print(f"📊 Episode {self.current_episode}, Step {self.current_step}, Total Reward: {self.total_reward:.2f}")
        
        # End of episode
        if done:
            print(f" Episode {self.current_episode} completed! Total reward: {self.total_reward:.2f}")
            
            # Update policy
            if len(self.agent.experience_buffer) >= BATCH_SIZE:
                self.agent.update_policy()
            
            # Save experience periodically
            if self.current_episode % 5 == 0:
                self.agent.save_to_csv()
            
            # Reset for next episode
            self.reset_episode()

# --- Global Environment Instance ---
car_env = CarEnvironment()

# --- Blender Integration ---
def training_frame_handler(scene):
    """Frame handler for training"""
    car_env.step()

class StartPPOTrainingOperator(bpy.types.Operator):
    bl_idname = "wm.start_ppo_training"
    bl_label = "Start PPO Car Training"
    bl_description = "Start PPO reinforcement learning for car control"
    
    def execute(self, context):
        global car_env
        car_env.training_active = True
        car_env.reset_episode()
        
        if training_frame_handler not in bpy.app.handlers.frame_change_pre:
            bpy.app.handlers.frame_change_pre.append(training_frame_handler)
        
        self.report({'INFO'}, "PPO Training Started!")
        print("Starting Car Training!")
        print("Experience will be saved to CSV files")
        return {'FINISHED'}

class StopPPOTrainingOperator(bpy.types.Operator):
    bl_idname = "wm.stop_ppo_training"
    bl_label = "Stop PPO Training"
    bl_description = "Stop PPO reinforcement learning"
    
    def execute(self, context):
        global car_env
        car_env.training_active = False
        
        # Save final experience
        car_env.agent.save_to_csv()
        
        if training_frame_handler in bpy.app.handlers.frame_change_pre:
            bpy.app.handlers.frame_change_pre.remove(training_frame_handler)
            
        self.report({'INFO'}, "PPO Training Stopped!")
        print(" Training Stopped!")
        return {'FINISHED'}

class SaveExperienceOperator(bpy.types.Operator):
    bl_idname = "wm.save_experience"
    bl_label = "Save Experience Buffer"
    bl_description = "Save current experience replay buffer to CSV"
    
    def execute(self, context):
        global car_env
        car_env.agent.save_to_csv()
        self.report({'INFO'}, "Experience Buffer Saved!")
        return {'FINISHED'}

def register():
    bpy.utils.register_class(StartPPOTrainingOperator)
    bpy.utils.register_class(StopPPOTrainingOperator)
    bpy.utils.register_class(SaveExperienceOperator)
    

def unregister():
    global car_env
    car_env.training_active = False
    
    if training_frame_handler in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(training_frame_handler)
    
    bpy.utils.unregister_class(StartPPOTrainingOperator)
    bpy.utils.unregister_class(StopPPOTrainingOperator)
    bpy.utils.unregister_class(SaveExperienceOperator)

# Auto-start training
register()
bpy.ops.wm.start_ppo_training()
