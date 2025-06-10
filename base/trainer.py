import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.criterion = nn.MSELoss(reduction='none')  # No reduction for prioritized replay
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none')  # Huber loss for stability

    def train_step(self, state, action, reward, next_state, done, target_model=None):
        """Standard training step - returns TD error for prioritized replay"""
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        done = torch.tensor(np.array(done), dtype=torch.bool)
        
        if len(state.shape) == 1:
            # Single sample
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        
        # Current Q values
        pred = self.model(state)
        target = pred.clone().detach()
        
        td_errors = []
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                if target_model:
                    next_q_values = target_model(next_state[idx]).detach()
                else:
                    next_q_values = self.model(next_state[idx]).detach()
                Q_new = reward[idx] + self.gamma * torch.max(next_q_values).item()
            
            action_idx = torch.argmax(action[idx]).item()
            old_q = target[idx][action_idx].item()
            target[idx][action_idx] = Q_new
            
            # Calculate TD error for prioritized replay
            td_error = abs(Q_new - old_q)
            td_errors.append(td_error)
        
        self.optimizer.zero_grad()
        loss = self.smooth_l1_loss(pred, target).mean()  # Use Huber loss
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        return td_errors[0] if len(td_errors) == 1 else td_errors

    def train_step_prioritized(self, states, actions, rewards, next_states, dones, target_model, weights):
        """Training step with prioritized experience replay"""
        state = torch.tensor(np.array(states), dtype=torch.float)
        next_state = torch.tensor(np.array(next_states), dtype=torch.float)
        action = torch.tensor(np.array(actions), dtype=torch.long)
        reward = torch.tensor(np.array(rewards), dtype=torch.float)
        done = torch.tensor(np.array(dones), dtype=torch.bool)
        weights = torch.tensor(weights, dtype=torch.float)
        
        # Current Q values
        current_q_values = self.model(state)
        
        # Get the Q values for the taken actions
        action_indices = torch.argmax(action, dim=1)
        current_q = current_q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            # Double DQN: use main network to select action, target network to evaluate
            next_q_values_main = self.model(next_state)
            next_actions = torch.argmax(next_q_values_main, dim=1)
            
            next_q_values_target = target_model(next_state)
            next_q = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            target_q = reward + (self.gamma * next_q * (~done))
        
        # Calculate TD errors
        td_errors = torch.abs(current_q - target_q).detach().cpu().numpy()
        
        # Weighted loss for prioritized experience replay
        loss = self.smooth_l1_loss(current_q, target_q)
        weighted_loss = (weights * loss).mean()
        
        self.optimizer.zero_grad()
        weighted_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        return td_errors