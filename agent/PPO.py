
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

class Community_actor_net(nn.Module):
    def __init__(self, community_dim, hidden_dim, community_action_dim):
        super(Community_actor_net, self).__init__()
        self.fc1= nn.Linear(community_dim, hidden_dim)
        self.fc2=nn.Linear(hidden_dim, hidden_dim)
        self.mu_layer = nn.Linear(hidden_dim, community_action_dim)
        self.log_std_layer = nn.Parameter(torch.ones(community_action_dim)*-1)


    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_layer(x)
        std=torch.exp(self.log_std_layer).expand_as(mu)

        return mu, std


class Critic_net(nn.Module):
    def __init__(self,community_state_dim,hidden_dim):
        super(Critic_net, self).__init__()
        self.fc1= nn.Linear(community_state_dim, hidden_dim)
        self.fc2= nn.Linear(hidden_dim, 1)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        return self.fc2(x)

class PPO_agent():
    def __init__(self,community_state_dim,hidden_dim,community_actor_dim,gamma):
        #1.神经网络搭建
        self.community_actor_net=Community_actor_net(community_state_dim,hidden_dim,community_actor_dim)
        self.critic_net=Critic_net(community_state_dim,hidden_dim)
        #2.配置优化器
        self.community_actor_net_optim=optim.Adam(self.community_actor_net.parameters(),lr=1e-5)
        self.critic_net_optim=optim.Adam(self.critic_net.parameters(),lr=2e-5)#community
        self.gamma=gamma

    def choose_cems_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        mu, std = self.community_actor_net(state)
        actions = []
        log_probs = []
        for i in range(mu.shape[1]):
            dist = Normal(mu[:, i], std[:, i])
            raw_a = dist.rsample()  # reparameterization trick
            # tanh squash
            a = torch.tanh(raw_a)
            # log_prob 修正（PPO 必须！）
            log_prob = dist.log_prob(raw_a) - torch.log(1 - a.pow(2) + 1e-6)
            if i == 0:
                final_a = 1 + 0.1 * a
            else:
                final_a = a
            actions.append(final_a)
            log_probs.append(log_prob)
        actions = torch.stack(actions, dim=1)
        log_probs = torch.stack(log_probs, dim=1)
        return actions.squeeze(0).detach().cpu().numpy(), log_probs.squeeze(0).detach()

    def compute_GAE(self,value,next_value,reward,done,lamda=0.95):
        lenth=reward.shape[0]
        GAE=torch.zeros_like(reward)
        gae=0
        for i in range(lenth-1,-1,-1):
            delta=self.gamma*next_value[i]*(1-done[i])+reward[i]-value[i]
            gae=self.gamma*lamda * gae+delta
            GAE[i]=gae
        return GAE

    def update(self,eps,community_state,community_actions,community_rewards,community_dones,community_history_next_state,community_history_lop_probs):
        '''历史数据预处理'''
        community_state=torch.tensor(np.array(community_state),dtype=torch.float32)
        community_actions = torch.tensor(np.array(community_actions), dtype=torch.float32)
        community_history_next_state=torch.tensor(np.array(community_history_next_state),dtype=torch.float32)
        community_rewards=torch.tensor(np.array(community_rewards),dtype=torch.float32).unsqueeze(1)
        community_dones=torch.tensor(np.array(community_dones),dtype=torch.float32).unsqueeze(1)
        community_history_lop_probs=torch.stack(community_history_lop_probs).detach()

        for _ in range(10):
            #critic网络更新
            value=self.critic_net(community_state)
            next_value=self.critic_net(community_history_next_state).detach()
            td_target=community_rewards+self.gamma*next_value*(1-community_dones)
            critic_loss=F.mse_loss(value,td_target)
            self.critic_net_optim.zero_grad()
            critic_loss.backward()
            self.critic_net_optim.step()
        #adv
            advantage=self.compute_GAE(value.detach(),next_value.detach(),community_rewards,community_dones)
            advantages=(advantage-advantage.mean())/(advantage.std()+1e-8)#A的归一化处理
            adv=advantages.detach()
        #actor更新
            mu,std=self.community_actor_net(community_state)
            community_actions = (community_actions - 1) / 0.1
            community_actions = community_actions.clamp(-0.999, 0.999)
            community_logp_new=[]
            entropy_sum = 0.0
            for i in range(mu.shape[1]):
                dist = Normal(mu[:, i], std[:, i])
                raw_a=torch.atanh(community_actions[:,i].clamp(-0.999, 0.999))
                log_prob = dist.log_prob(raw_a)-torch.log(1-community_actions[:,i]**2+1e-6)
                community_logp_new.append(log_prob)
                entropy_sum += dist.entropy()  # 添加熵
            community_logp_new=torch.stack(community_logp_new, dim=1)
            community_logp_sum=community_logp_new.sum(dim=1, keepdim=True)
            community_logp_old_sum=community_history_lop_probs.sum(dim=1, keepdim=True)
            ratio = torch.exp(community_logp_sum - community_logp_old_sum)
            com_surr1=ratio * adv
            com_surr2= torch.clamp(ratio, 1 - eps, 1 + eps) * adv
            com_loss=-torch.min(com_surr1, com_surr2).mean()-0.02*entropy_sum.mean()
            self.community_actor_net_optim.zero_grad()
            com_loss.backward()
            self.community_actor_net_optim.step()

























