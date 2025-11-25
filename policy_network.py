"""
Policy Network - Simplified MLP-based policy for Query Mutation
Standalone copy adapted from RLbreaker
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def init(module, weight_init, bias_init, gain=1):
    """Helper to initialize network weights"""
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Normalizer:
    """Online normalization of observations"""
    def __init__(self, in_size, device='cpu', dtype=torch.float):
        self.mean = torch.zeros((1, in_size), device=device, dtype=dtype)
        self.std = torch.ones((1, in_size), device=device, dtype=dtype)
        self.eps = 1e-5
        self.device = device
        self.count = self.eps

    def update_stats(self, data):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float().to(self.device)
        data = data.to('cpu')
        batch_mean = data.mean(0, keepdim=True)
        batch_var = data.var(0, keepdim=True)
        batch_count = data.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = torch.square(self.std) * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.std = torch.sqrt(new_var)
        self.count = tot_count

    def normalize(self, val):
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val).to(self.device)
        std = torch.clamp(self.std, self.eps)
        return (val - self.mean.to(val.device)) / std.to(val.device)


class FixedCategorical(torch.distributions.Categorical):
    """Categorical distribution for discrete action spaces"""
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

    def entropy(self):
        return super().entropy()


class Categorical(nn.Module):
    """Categorical action distribution"""
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        
        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class MLPBase(nn.Module):
    """MLP base network for policy"""
    def __init__(self, num_inputs, device, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__()
        
        self.normalizer = Normalizer(num_inputs, device=device)
        self._recurrent = recurrent
        self._hidden_size = hidden_size

        if recurrent:
            self.gru = nn.GRU(num_inputs, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU())
        
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def forward(self, inputs, rnn_hxs, masks):
        x = self.normalizer.normalize(inputs)
        
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            N = hxs.size(0)
            T = int(x.size(0) / N)
            x = x.view(T, N, x.size(1))
            masks = masks.view(T, N)
            
            has_zeros = ((masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu())
            if has_zeros.dim() == 0:
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()
            
            has_zeros = [0] + has_zeros + [T]
            hxs = hxs.unsqueeze(0)
            outputs = []
            
            for i in range(len(has_zeros) - 1):
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))
                outputs.append(rnn_scores)
            
            x = torch.cat(outputs, dim=0)
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class Policy(nn.Module):
    """Actor-Critic Policy Network"""
    def __init__(self, obs_shape, action_space, base_kwargs=None):
        super(Policy, self).__init__()
        
        if base_kwargs is None:
            base_kwargs = {}
        
        device = base_kwargs.get('device', 'cpu')
        
        # Only support 1D observation space (MLP)
        assert len(obs_shape) == 1, "Only 1D observations supported"
        
        hidden_size = base_kwargs.get('hidden_size', 64)
        recurrent = base_kwargs.get('recurrent', False)
        
        self.base = MLPBase(obs_shape[0], device, recurrent=recurrent, hidden_size=hidden_size)
        
        # Only support Discrete action space
        assert action_space.__class__.__name__ == "Discrete", "Only Discrete actions supported"
        num_outputs = action_space.n
        self.dist = Categorical(self.base.output_size, num_outputs)

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs
