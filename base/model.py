import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, hidden3=None, output_size=4):
        super().__init__()
        
        # Enhanced architecture với Layer Normalization thay vì Batch Normalization
        self.input_size = input_size
        self.output_size = output_size
        
        # Feature extraction layers
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.LayerNorm(hidden1),  # LayerNorm thay vì BatchNorm
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden1, hidden2),
            nn.LayerNorm(hidden2),  # LayerNorm thay vì BatchNorm
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Dueling DQN với improved architecture
        # Value stream - estimates V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden2, 128),
            nn.LayerNorm(128),  # LayerNorm thay vì BatchNorm
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream - estimates A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden2, 128),
            nn.LayerNorm(128),  # LayerNorm thay vì BatchNorm
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        
        # Noisy layers cho better exploration (optional)
        self.use_noisy = False
        if hidden3:  # Use hidden3 parameter to enable noisy layers
            self.use_noisy = True
            self.noisy_value = NoisyLinear(64, 1)
            self.noisy_advantage = NoisyLinear(64, output_size)
        
        # Better weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Xavier initialization for better gradient flow
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)
        elif isinstance(module, nn.LayerNorm):  # LayerNorm initialization
            module.weight.data.fill_(1.0)
            module.bias.data.fill_(0.0)

    def forward(self, x):
        # Handle both single samples and batches - LayerNorm works with any batch size
        if x.dim() == 1:
            x = x.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
        
        # Extract features
        features = self.feature_layer(x)
        
        # Dueling streams
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Dueling DQN formula: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        advantage_mean = advantages.mean(dim=1, keepdim=True)
        q_values = values + (advantages - advantage_mean)
        
        # Return to original shape if single sample
        if single_sample:
            q_values = q_values.squeeze(0)
            
        return q_values
    
    def reset_noise(self):
        """Reset noise in noisy layers if enabled"""
        if self.use_noisy:
            if hasattr(self, 'noisy_value'):
                self.noisy_value.reset_noise()
            if hasattr(self, 'noisy_advantage'):
                self.noisy_advantage.reset_noise()

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_path = os.path.join(model_folder_path, file_name)
        
        # Save both model state and architecture info
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'output_size': self.output_size,
            'architecture': 'Enhanced_Dueling_DQN'
        }
        torch.save(checkpoint, file_path)
        print(f"Enhanced model saved to {file_path}")

class NoisyLinear(nn.Module):
    """Noisy Linear layer for better exploration"""
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Noise tensors (not learnable)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / (self.in_features ** 0.5)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / (self.in_features ** 0.5))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / (self.out_features ** 0.5))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)