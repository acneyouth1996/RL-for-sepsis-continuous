"""
The classes and methods in this file are derived or pulled directly from https://github.com/sfujim/BCQ/tree/master/discrete_BCQ
which is a discrete implementation of BCQ by Scott Fujimoto, et al. and featured in the following 2019 DRL NeurIPS workshop paper:
@article{fujimoto2019benchmarking,
  title={Benchmarking Batch Deep Reinforcement Learning Algorithms},
  author={Fujimoto, Scott and Conti,
      Edoardo and Ghavamzadeh, Mohammad and Pineau, Joelle},
  journal={arXiv preprint arXiv:1910.01708},
  year={2019}
}

============================================================================================================================
This code is provided under the MIT License and is meant to be helpful, but WITHOUT ANY WARRANTY;

November 2020 by Taylor Killian and Haoran Zhang; University of Toronto + Vector Institute
============================================================================================================================
Notes:

"""

import argparse
from cgi import test
import copy
from email import policy
import importlib
import json
import os
import re


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from torch.utils.tensorboard import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt
#import tkinter
import seaborn as sns
#matplotlib.use('TkAgg')

class OUNoise:
    def __init__(self, action_dimension, dt=0.01, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.dt = dt
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu
        #self.state.to(self.device)

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.random.randn(len(x)) * np.sqrt(self.dt)
        self.state = x + dx
        return self.state


class Actor(nn.Module):
    """
    The mu network in https://spinningup.openai.com/en/latest/algorithms/ddpg.html
    the network that output the action which gives the max Q value
    """

    def __init__(self, state_dim, action_dim, num_nodes=8):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, num_nodes)
        
        self.l2 = nn.Linear(num_nodes, num_nodes)
        
        self.l3 = nn.Linear(num_nodes, action_dim)

       
        
        

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.sigmoid(self.l3(x))
       
        return x


class Critic(nn.Module):
    """
    The standard Q network
    """

    def __init__(self, state_dim, action_dim, num_nodes=32):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, num_nodes)
        
        self.l2 = nn.Linear(num_nodes, num_nodes)
        
        self.l3 = nn.Linear(num_nodes, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        # since the reward are the final outcome, it is either 1 or -1 at the final timestep, 0 other places,
        # so the Q value would not exceed [-1, 1], we clip the Q value output
        #x = torch.clamp(x, -6.4, 6.4)
        return x


class DDPG(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        writer,
        discount=0.9,
        optimizer="Adam",
        actor_optimizer_parameters={},
        critic_optimizer_parameters={},
        target_update_frequency=1e2,
        tau=0.01
    ):
        super(DDPG, self).__init__()
        self.device = device
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = getattr(torch.optim, optimizer)(
            self.actor.parameters(), **actor_optimizer_parameters)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = getattr(torch.optim, optimizer)(
            self.critic.parameters(), **critic_optimizer_parameters)
        self.discount = discount
        self.target_update_frequency = target_update_frequency
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0
        self.writer = writer
        self.tau = tau
        self.noise = OUNoise(2)
        self.loss = nn.MSELoss(reduction = 'None')

    def cal_action(self, actor, state, noise):
        raw_action = actor(state)
        #noise = torch.Tensor(noise).to(self.device)
        #action = raw_action + noise
        return raw_action

    def reform_reward(self, reward, score, next_score, done):
        
        
        sofa_reward = -0.125 * (next_score[:, 0] - score[:, 0])
        sofa_reward = sofa_reward.unsqueeze(1)
        #print(sofa_reward)
        #print(sofa_reward)
        #reward = 10 * reward
        #final_reward = sofa_reward + reward
        #print('r',reward.shape)
        #print(done.shape)
        #print((reward * done).shape)
        #final_reward = reward.squeeze(1) * done.squeeze(1) + sofa_reward
        return sofa_reward

        
    def train(self, replay_buffer, train_iters):
        # Sample replay buffer
        self.noise.reset()
        noise = self.noise.noise()
        self.num_critic_update_iteration += 1
        self.num_actor_update_iteration += 1
        state, action_c, next_state, reward, scores, next_scores, outcome,  not_done = replay_buffer.sample()
        #reward = torch.ones_like(reward) * outcome
        reward = self.reform_reward(reward, scores, next_scores, not_done)
        
        # Compute the target Q value
        with torch.no_grad():
            
            action_rl = self.cal_action(self.actor_target, next_state, noise)
            
            target_Q = self.critic_target(next_state, action_rl)
            #print('f',target_Q.shape)
            target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action_c)
        
        # Compute critic loss
        #q_loss = F.smooth_l1_loss(current_Q, target_Q)

        critic_loss = F.mse_loss(current_Q, target_Q)
        
        self.writer.add_scalar('Loss/critic_loss', critic_loss,
                               self.num_critic_update_iteration)


        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

         # Compute actor loss
        actor_loss = -self.critic(state, self.cal_action(self.actor, state, noise)).mean() 
        self.writer.add_scalar('Loss/actor_loss', actor_loss,
                               self.num_actor_update_iteration)

        self.writer.add_scalar('Loss/iv rl action', action_rl[:,0].mean(),
                               self.num_actor_update_iteration)

        self.writer.add_scalar('Loss/vc rl action', action_rl[:,1].mean(),
                               self.num_actor_update_iteration)

        self.writer.add_scalar('Loss/iv cl action', action_c[:,0].mean(),
                               self.num_actor_update_iteration)

        self.writer.add_scalar('Loss/vc cl action', action_c[:,1].mean(),
                               self.num_actor_update_iteration)

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        if train_iters+1 %self.target_update_frequency == 0:

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss, critic_loss, target_Q, current_Q






def train_DDPG(replay_buffer, test_replay_buffer, state_dim, action_dim,  device, parameters, writer):
    """
    one thing to note that the state dim is not the dim of raw data, it is the dimension output by auto-encoder
    which is hidden size used in experiments script
    """

    buffer_dir = parameters['buffer_dir']
    test_buffer_dir = parameters['test_buffer_dir']

    # Initialize and load policy
    policy = DDPG(
        state_dim,
        action_dim,
        device,
        writer,
        parameters["discount"],
        parameters["optimizer"],
        parameters["actor_optimizer_parameters"],
        parameters["critic_optimizer_parameters"],
        parameters["target_update_freq"],
        parameters["tau"]
    )
    print(state_dim)
    print(action_dim)

    # Load replay buffer
    replay_buffer.load(buffer_dir, bootstrap=True)
    test_replay_buffer.load(test_buffer_dir, bootstrap=True)

    evaluations = []
    training_iters = 0

    # eval freq is essentially how often we write stuff to tensorboard
    while training_iters < parameters["max_timesteps"]:
        
        for _ in range(int(parameters["eval_freq"])):
            actor_l, critic_l, targ_q, cur_q = policy.train(replay_buffer, training_iters)

        training_iters += int(parameters["eval_freq"])
        print(f"Training iterations: {training_iters}")

        writer.add_scalar('Current Q value', torch.mean(targ_q), training_iters)

    direct_eval(policy, test_replay_buffer,  [0, 1], device, parameters)
    plot_action_iv(policy, replay_buffer,  parameters)
    plot_action_vc(policy, replay_buffer,  parameters)
    u_plot(policy, test_replay_buffer, device, parameters)




def est_mort(rl_policy, replay_buffer,  vc_range, device, parameters):
    """
    eval_policy: the policy to be evaluated
    replay_buffer: test replay buffer with bootstrap == True
    iv_range and vc_range: the min and max actions for the 2 actions, used to create uniform random policy
    """
    eval_iters = 0
    Q_e, Q_c, Q_r = 0, 0, 0
    total_correct = 0
    actual_mort = []
    pred_mort = []
    while eval_iters < parameters["eval_steps"]:
        state, action_c, next_state, reward, scores, next_scores, outcome, done = replay_buffer.sample()
        # print(state.sum())
        #action_c = action_c[:,1].reshape(-1, 1)
        action_rl = rl_policy.actor(state)
        batch_size = action_c.shape[0]
        iv_rand = torch.FloatTensor(batch_size, 1).uniform_(
            vc_range[0], vc_range[1])
        vc_rand = torch.FloatTensor(batch_size, 1).uniform_(
            vc_range[0], vc_range[1])
        action_rd = torch.cat((iv_rand ,vc_rand), axis=1).to(device)
        # print(action_rd.shape)
        Q_estimate = rl_policy.critic(state, action_rl).cpu()
        #print(Q_estimate)
        Q_clinician = rl_policy.critic(state, action_c).cpu()
        Q_random = rl_policy.critic(state, action_rd).cpu()
        del action_rd  # release

        actual_outcome = torch.where(outcome== -1, 1, 0)
        pred_outcome = ((Q_clinician+1)/2).cpu().detach().numpy()

        Q_estimate = torch.where(Q_estimate < 0, 1, 0)

        Q_clinician = torch.where(Q_clinician < 0, 1, 0) # death 1, survive 0
        Q_random = torch.where(Q_random < 0, 1, 0)

         # death 1, survive 0
        
        
        pred_mort.append(Q_clinician.cpu().detach().numpy())
        actual_mort.append(actual_outcome.cpu().numpy())
        

        pred_correct = (Q_clinician == actual_outcome.cpu()).sum()
        total_correct+= pred_correct


        eval_mort = Q_estimate.sum()
        
        Q_e += eval_mort
        cli_mort = Q_clinician.sum()
        Q_c += cli_mort
        ran_mort = Q_random.sum()
        Q_r += ran_mort

        eval_iters += 1
    
    actual_mort = np.concatenate(actual_mort, axis=None)
    pred_mort = np.concatenate(pred_mort, axis=None) 

    actual_mort_idx = np.where(actual_mort==1)[0]
    actual_mort_true = actual_mort[actual_mort_idx]
    pred_mort_true = pred_mort[actual_mort_idx]


    print('Q estimate', Q_e/(eval_iters*batch_size))
    print('Q clinician', Q_c/(eval_iters*batch_size))
    print('Q random', Q_r/(eval_iters*batch_size))
    print('Observed mortality prediction accuracy', total_correct/(eval_iters*batch_size))
    print('death accuracy', (actual_mort_true==pred_mort_true).sum()/len(actual_mort_true))

def direct_eval(rl_policy, replay_buffer,  vc_range, device, parameters):
    """
    eval_policy: the policy to be evaluated
    replay_buffer: test replay buffer with bootstrap == True
    iv_range and vc_range: the min and max actions for the 2 actions, used to create uniform random policy
    """
    eval_iters = 0
    Q_e, Q_c, Q_r, Q_z = 0, 0, 0, 0
    
    rl_policy.critic.eval()
    rl_policy.actor.eval()
    rl_policy.critic_target.eval()
    rl_policy.actor_target.eval()
    
    while eval_iters < parameters["eval_steps"]:
        state, action_c, next_state, reward, scores, next_scores, outcome, done = replay_buffer.sample()
        
        action_rl = rl_policy.actor_target(state)
        batch_size = action_c.shape[0]
        iv_rand = torch.FloatTensor(batch_size, 1).uniform_(
            vc_range[0], vc_range[1])
        vc_rand = torch.FloatTensor(batch_size, 1).uniform_(
            vc_range[0], vc_range[1])
        action_rd = torch.cat((iv_rand ,vc_rand), axis=1).to(device)
        action_zero = torch.zeros_like(action_c).to(device)
        Q_estimate = rl_policy.critic(state, action_rl).cpu()
       
        Q_clinician = rl_policy.critic(state, action_c).cpu()
        Q_random = rl_policy.critic(state, action_rd).cpu()
        Q_zero = rl_policy.critic(state, action_zero).cpu()
        del action_rd  # release
        del action_zero
        
        Q_e += Q_estimate.sum()
       
        Q_c += Q_clinician.sum()
       
        Q_r += Q_random.sum()

        Q_z += Q_zero.sum()

        eval_iters += 1
    
    

    print('Q estimate', Q_e/(eval_iters*batch_size))
    print('Q clinician', Q_c/(eval_iters*batch_size))
    print('Q random', Q_r/(eval_iters*batch_size))
    print('Q zero', Q_z/(eval_iters*batch_size))
    





def plot_action_iv(rl_policy, replay_buffer,  parameters):
    """
    3d scatter plot of different policies, x and y axis are actions (vc and iv)
    z axis are sofa score
    """
    eval_iters = 0
    fig, axes = plt.subplots(1, 4)
    xs_rl = []
    xs_cl = []

    rl_policy.critic.eval()
    rl_policy.actor.eval()
    rl_policy.critic_target.eval()
    rl_policy.actor_target.eval()
    while eval_iters < parameters["eval_steps"]:
        state, action_c, next_state, reward, scores,next_scores, outcome, done = replay_buffer.sample()
        action_rl =  rl_policy.actor_target(next_state)
        iv_rl = action_rl[:, 0]
        
        xs_rl.append(iv_rl.detach().cpu().numpy())
        xs_cl.append(action_c[:,0].cpu().numpy())
        
        eval_iters+=1

    
    xs_rl = np.concatenate(xs_rl, axis=None)
    xs_rl = xs_rl *10000

    xs_cl = np.concatenate(xs_cl, axis=None)
    xs_cl = xs_cl * 10000
    

    print('rl iv mean',xs_rl.mean())
    print('cl iv mean',xs_cl.mean())

    
    sns.set_style('whitegrid')
    

    sns.distplot(xs_cl, color='y', kde_kws={"clip":(0,50)}, hist_kws={"range":(0,50)}, ax=axes[0])
    sns.distplot(xs_rl, color='r', kde_kws={"clip":(0,50)}, hist_kws={"range":(0,50)}, ax=axes[0])

    sns.distplot(xs_cl, color='y', kde_kws={"clip":(50,152)}, hist_kws={"range":(50,152)}, ax=axes[1])
    sns.distplot(xs_rl, color='r', kde_kws={"clip":(50,152)}, hist_kws={"range":(50,152)}, ax=axes[1])

    sns.distplot(xs_cl, color='y', kde_kws={"clip":(152,500)}, hist_kws={"range":(152,500)}, ax=axes[2])
    sns.distplot(xs_rl, color='r', kde_kws={"clip":(152,500)}, hist_kws={"range":(152,500)}, ax=axes[2])

    sns.distplot(xs_cl, color='y', kde_kws={"clip":(500,10000)}, hist_kws={"range":(500,10000)}, ax=axes[3])
    sns.distplot(xs_rl, color='r', kde_kws={"clip":(500,10000)}, hist_kws={"range":(500, 10000)}, ax=axes[3])
   
    plt.show()


def plot_action_vc(rl_policy, replay_buffer,   parameters):
    """
    3d scatter plot of different policies, x and y axis are actions (vc and iv)
    z axis are sofa score
    """
    eval_iters = 0
    
    fig, axes = plt.subplots(1, 4)
   
    ys_rl = []
    ys_cl = []
    rl_policy.critic.eval()
    rl_policy.actor.eval()
    rl_policy.critic_target.eval()
    rl_policy.actor_target.eval()
    while eval_iters < parameters["eval_steps"]:
        state, action_c, next_state, reward, scores,next_scores, outcome, done = replay_buffer.sample()
        
        action_rl =  rl_policy.actor_target(next_state)
        
        vc_rl = action_rl[:, 1]
        ys_rl.append(vc_rl.detach().cpu().numpy())
        ys_cl.append(action_c[:,1].cpu().numpy())
        
        
        eval_iters+=1

    
    ys_rl = np.concatenate(ys_rl, axis=None)
    ys_rl = ys_rl * 189.076
    ys_cl = np.concatenate(ys_cl, axis=None)
    ys_cl = ys_cl * 189.076
   
    print('rl vc mean',ys_rl.mean())
    print('cl vc mean',ys_cl.mean())

    
    sns.set_style('whitegrid')
    
    sns.distplot(ys_cl, color='g', kde_kws={"clip":(0,0.2)}, hist_kws={"range":(0,0.2)}, ax=axes[0])
    sns.distplot(ys_rl, color='b', kde_kws={"clip":(0,0.2)}, hist_kws={"range":(0,0.2)}, ax=axes[0])

    sns.distplot(ys_cl, color='g', kde_kws={"clip":(0.2,0.45)}, hist_kws={"range":(0.2,0.45)}, ax=axes[1])
    sns.distplot(ys_rl, color='b', kde_kws={"clip":(0.2,0.45)}, hist_kws={"range":(0.2,0.45)}, ax=axes[1])

    sns.distplot(ys_cl, color='g', kde_kws={"clip":(0.45,1.91)}, hist_kws={"range":(0.45,1.91)}, ax=axes[2])
    sns.distplot(ys_rl, color='b', kde_kws={"clip":(0.45,1.91)}, hist_kws={"range":(0.45,1.91)}, ax=axes[2])

    sns.distplot(ys_cl, color='g', kde_kws={"clip":(1.91,190)}, hist_kws={"range":(1.91,190)}, ax=axes[3])
    sns.distplot(ys_rl, color='b', kde_kws={"clip":(1.91,190)}, hist_kws={"range":(1.91, 190)}, ax=axes[3])
    
    plt.show()

def u_plot(rl_policy, replay_buffer, device, parameters):
    from scipy.stats import sem
    eval_iters = 0
    iv_diffs, vc_diffs = [], []
    death =[]
    while eval_iters < parameters["eval_steps"]:
        state, action_c, next_state, reward, scores, next_scores, outcome, done = replay_buffer.sample()
        #action_c = action_c[:,1].reshape(-1, 1)
        outcome = torch.where(outcome == -1, 0, 1)
        # print(state.sum())
        action_rl = rl_policy.actor_target(state)
        batch_size = action_c.shape[0]
        iv_d = action_rl[:,0] - action_c[:,0]
        vc_d = action_rl[:,1] - action_c[:,1]
        #vc_d = action_rl - action_c
        iv_diffs.append(iv_d.cpu().detach().numpy())
        vc_diffs.append(vc_d.cpu().detach().numpy())
        death.append(outcome.cpu().detach().numpy())
        eval_iters+=1
    
    print('data ready')
    iv_diffs = np.concatenate(iv_diffs, axis=None)
    vc_diffs = np.concatenate(vc_diffs, axis = None)
    death = np.concatenate(death, axis = None)
    
        

    mort_vc, mort_iv = [],[]
    bin_vc, bin_iv = [], []
    std_vc, std_iv = [], []
    i = 1.0
    while i >=-1.0:
        idx_vc = np.where((vc_diffs>i-0.05) & (vc_diffs<i+0.05))[0]
        idx_iv = np.where((iv_diffs>i-0.05) & (iv_diffs<i+0.05))[0]
        death_vc = death[idx_vc]
        death_iv = death[idx_iv]
        
        death_mean_vc = (len(death_vc)-death_vc.sum())/len(death_vc)
        death_mean_iv = (len(death_iv)-death_iv.sum())/len(death_iv)
        
        death_se_vc = sem(death_mean_vc)
        death_se_iv = sem(death_mean_iv)

        mort_vc.append(death_mean_vc)
        mort_iv.append(death_mean_iv)
        bin_vc.append(i)
        bin_iv.append(i)
        std_vc.append(death_se_vc)
        std_iv.append(death_se_iv)
        i-=0.1

    mort_vc = np.array(mort_vc)
    mort_iv = np.array(mort_iv)
    bin_vc = np.array(bin_vc)
    bin_iv = np.array(bin_iv)
    std_vc = np.array(std_vc)
    std_iv = np.array(std_iv)

    fig, axes = plt.subplots(1, 2)

    axes[0].plot(bin_iv, mort_iv, color='r')
    axes[0].fill_between(bin_iv, mort_iv - 1*std_iv,  mort_iv + 1*std_iv, color='tomato')

    axes[1].plot(bin_vc, mort_vc, color='g')
    axes[1].fill_between(bin_vc, mort_vc - 1*std_vc,  mort_vc + 1*std_vc, color='green')
    plt.show()
    




    


    
