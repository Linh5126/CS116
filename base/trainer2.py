import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done, target_model=None):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        done = torch.tensor(np.array(done), dtype=torch.bool)

        if len(state.shape) == 1:
            # Single sample
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)

        # 1. Q values predicted for current state
        pred_q = self.model(state)

        # 2. Get indices of the actions taken
        action_indices = torch.argmax(action, dim=1, keepdim=True)

        # 3. Gather predicted Q values for those actions
        pred = pred_q.gather(1, action_indices).squeeze(1)

        # 4. Compute target Q values using Double DQN
        with torch.no_grad():
            if target_model:
                # Double DQN: Use current model to select actions, target model to evaluate
                next_action_indices = self.model(next_state).argmax(dim=1, keepdim=True)
                next_q_target = target_model(next_state).gather(1, next_action_indices).squeeze(1)
            else:
                # Standard DQN fallback: just take max Q from current model
                next_q_target = self.model(next_state).max(1)[0]

            # Compute Q target: r + γ * Q_target(s', argmax_a Q(s',a)) * (1 - done)
            target = reward + self.gamma * next_q_target * (~done)

        # 5. Compute loss and backpropagate
        loss = self.criterion(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping để tránh exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()

        # Return TD error for prioritized replay
        td_error = torch.abs(pred - target).detach().numpy()
        if len(td_error) == 1:
            return td_error[0]
        return td_error

    def train_step_prioritized(self, states, actions, rewards, next_states, dones, target_model, weights):
        """Enhanced training step for prioritized experience replay"""
        state = torch.tensor(np.array(states), dtype=torch.float)
        next_state = torch.tensor(np.array(next_states), dtype=torch.float)
        action = torch.tensor(np.array(actions), dtype=torch.long)
        reward = torch.tensor(np.array(rewards), dtype=torch.float)
        done = torch.tensor(np.array(dones), dtype=torch.bool)
        weights = torch.tensor(weights, dtype=torch.float)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)
            weights = weights.unsqueeze(0)

        # 1. Current Q values
        pred_q = self.model(state)
        action_indices = torch.argmax(action, dim=1, keepdim=True)
        pred = pred_q.gather(1, action_indices).squeeze(1)

        # 2. Target Q values using Double DQN
        with torch.no_grad():
            if target_model:
                # Double DQN
                next_action_indices = self.model(next_state).argmax(dim=1, keepdim=True)
                next_q_target = target_model(next_state).gather(1, next_action_indices).squeeze(1)
            else:
                next_q_target = self.model(next_state).max(1)[0]

            target = reward + self.gamma * next_q_target * (~done)

        # 3. Compute weighted loss (importance sampling)
        td_errors = target - pred
        weighted_loss = (weights * td_errors.pow(2)).mean()

        # 4. Backpropagation
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Return TD errors for priority updates
        return torch.abs(td_errors).detach().numpy()