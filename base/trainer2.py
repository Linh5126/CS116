import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done, target_model=None):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        done = torch.tensor(np.array(done), dtype=torch.bool)

        if len(state.shape) == 1:
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
                # Use current model to get action indices
                next_action_indices = self.model(next_state).argmax(dim=1, keepdim=True)
                # Use target model to get Q-values at those actions
                next_q_target = target_model(next_state).gather(1, next_action_indices).squeeze(1)
            else:
                # Standard DQN: just take max Q from model
                next_q_target = self.model(next_state).max(1)[0]

            # Compute Q target: r + γ * maxQ’(s’, a’) * (1 - done)
            target = reward + self.gamma * next_q_target * (~done)

        # 5. Compute loss and backpropagate
        loss = self.criterion(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()