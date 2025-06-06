import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done, target_model=None):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.float)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        done = torch.tensor(np.array(done), dtype=torch.bool)

        pred = self.model(state)
        target = pred.clone().detach()

        for idx in range(len(done)):
            if target_model:
                next_q_values = target_model(next_state[idx])
            else:
                next_q_values = self.model(next_state[idx])

            Q_new = reward[idx]
            if not done[idx]:
                Q_new += self.gamma * torch.max(next_q_values).item()

            action_idx = torch.argmax(action[idx]).item()
            target[idx][action_idx] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()
