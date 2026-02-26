import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
class Actor_net(nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(Actor_net, self).__init__()
        self.fc1= nn.Linear(state_dim, hidden_dim)
        self.fc2=nn.Linear(hidden_dim, hidden_dim)
        self.mu_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Parameter(torch.ones(action_dim)*-1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_layer(x)
        log_std = torch.clamp(self.log_std_layer, -2, 2)
        std=torch.exp(self.log_std_layer).expand_as(mu)

        return mu, std

class Critic_Net(nn.Module):
    def __init__(self,state_dim,hidden_dim):
        super(Critic_Net, self).__init__()
        self.fc1= nn.Linear(state_dim, hidden_dim)
        self.fc2=nn.Linear(hidden_dim, 1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class PPOHEMS_agent():
    def __init__(self,state_dim,hidden_dim,action_dim,gamma):
        self.actor_net = Actor_net(state_dim,hidden_dim,action_dim)
        self.hems_critic_net=Critic_Net(state_dim,hidden_dim)
        self.actor_net_optimizer = optim.Adam(self.actor_net.parameters(), lr=1e-5)
        self.critic_net_optimizer = optim.Adam(self.hems_critic_net.parameters(), lr=2e-5)
        self.gamma=gamma

    def choose_hems_action(self,  state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        mu, std = self.actor_net(state)
        actions = []
        log_probs = []
        for i in range(mu.shape[1]):
            # ===== Bernoulli (MW, DIS) =====
            if i in [0, 1]:
                p = torch.sigmoid(mu[:, i])
                dist = torch.distributions.Bernoulli(p)
                a = dist.sample()
                log_prob = dist.log_prob(a)
            # ===== Normal (continuous) =====
            else:
                dist = Normal(mu[:, i], std[:, i])
                raw_a = dist.rsample()
                a = torch.tanh(raw_a)
                log_prob = dist.log_prob(raw_a) - torch.log(1 - a ** 2 + 1e-6)
            actions.append(a)
            log_probs.append(log_prob)
        actions = torch.stack(actions, dim=1)  # shape [1, act_dim]
        log_probs = torch.stack(log_probs, dim=1)  # shape [1, act_dim]
        return actions.squeeze(0).detach().cpu().numpy(), log_probs.squeeze(0).detach()

    def comupte_GAE(self,value,next_value,reward,done,lammda=0.95):
        lenth=reward.shape[0]
        GAE = torch.zeros_like(reward)
        gae=0
        for i in range(lenth-1,-1,-1):
            delta=reward[i]+self.gamma*next_value[i]*(1-done[i])-value[i]
            gae=lammda*self.gamma *gae + delta
            GAE[i]=gae
        return GAE

    def compute_log_probs(self, mu, std, actions, bernoulli_idx=[0, 1]):
        log_probs = []

        for i in range(actions.shape[1]):
            # Bernoulli action (0/1)
            if i in bernoulli_idx:
                p = torch.sigmoid(mu[:, i])
                dist = torch.distributions.Bernoulli(probs=p)
                log_prob = dist.log_prob(actions[:, i])
            # Continuous action (tanh-squashed Normal)
            else:
                dist = Normal(mu[:, i], std[:, i])
                raw_action = torch.atanh(actions[:, i].clamp(-0.999, 0.999))
                log_prob = dist.log_prob(raw_action) - torch.log(1 - actions[:, i] ** 2 + 1e-6)

            log_probs.append(log_prob)

        log_probs = torch.stack(log_probs, dim=1)  # [B, act_dim]
        return log_probs.sum(dim=1, keepdim=True)  # [B, 1]

    def hems_update(self,eps,state,action,reward,next_state,done,history_log_prob):
        state=torch.tensor(np.array(state), dtype=torch.float32)
        action=torch.tensor(np.array(action), dtype=torch.float32)
        reward=torch.tensor(np.array(reward), dtype=torch.float32).unsqueeze(1)
        next_state=torch.tensor(np.array(next_state), dtype=torch.float32)
        done=torch.tensor(np.array(done), dtype=torch.float32).unsqueeze(1)
        history_log_prob=torch.stack(history_log_prob).detach()


        for _ in range(10):
            value=self.hems_critic_net(state)
            next_value=self.hems_critic_net(next_state)

            td_target=self.gamma *next_value *(1-done)+reward
            td_target=td_target.detach()
            critic_loss=F.mse_loss(value, td_target)

            self.critic_net_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_net_optimizer.step()

            advantage=self.comupte_GAE(value.detach(),next_value.detach(),reward,done)
            advantages=(advantage-advantage.mean())/(advantage.std()+1e-8)
            advantages=advantages.detach()

            mu,std=self.actor_net(state)
            log_new_prob_sum=self.compute_log_probs(mu,std,action)
            log_old_prob_sum=history_log_prob.sum(dim=1, keepdim=True)
            ratio=torch.exp(log_new_prob_sum-log_old_prob_sum)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
            entropy_sum = 0.0
            for i in range(mu.shape[1]):
                if i in [0, 1]:
                    p = torch.sigmoid(mu[:, i])
                    dist = torch.distributions.Bernoulli(p)
                else:
                    dist = Normal(mu[:, i], std[:, i])
                entropy_sum += dist.entropy()
            loss = -torch.min(surr1, surr2).mean() - 0.02 * entropy_sum.mean()
            self.actor_net_optimizer.zero_grad()
            loss.backward()
            self.actor_net_optimizer.step()
