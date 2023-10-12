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
from operator import ne
import os
import re
from scipy.stats import sem


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from torch.utils.tensorboard import SummaryWriter
# import matplotlib
# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
#plt.rc('font', size=14)  
import seaborn as sns


def custom_loss(output, target):
    temp = torch.abs(target-output).double()
    #print(type(temp)) 
    loss = torch.where(temp> 0.7, temp, 0.0)
    return loss.mean()



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim,num_nodes=32):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, num_nodes)
        self.l2 = nn.Linear(num_nodes, num_nodes)
        self.l3 = nn.Linear(num_nodes, action_dim)
        
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        
        #return torch.sigmoid(a)
        #return self.l3(a)
        return (torch.tanh(self.l3(a))+1)/2
        
 


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim,  num_nodes = 32):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim +1, num_nodes)
        self.l2 = nn.Linear(num_nodes, num_nodes)
        self.l3 = nn.Linear(num_nodes, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim+ 1, num_nodes)
        self.l5 = nn.Linear(num_nodes, num_nodes)
        self.l6 = nn.Linear(num_nodes, 1)

       


    def forward(self, state, action, sofa):
        
        sa = torch.cat([state, action, sofa], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        #q1 = torch.clamp(q1, -1, 1)
        #q2 = torch.clamp(q2, -1, 1)
        
        
       
        return q1, q2


    def Q1(self, state, action, sofa):
        sa = torch.cat([state, action, sofa], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        #q1 = torch.clamp(q1, -1, 1)
        return q1





class TD3(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        writer,
        discount=0.99,
        tau=0.005,
        policy_freq=10,
        
        

        
    ):
        super(TD3, self).__init__()
        
        
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

        self.discount = discount
        self.tau = tau
        self.policy_freq = policy_freq
        self.total_it = 0
        self.device = device
        self.writer = writer
        
        self.loss = nn.MSELoss()
        

    def cal_action(self, state):
        raw_action = self.actor(state)
        return raw_action

    def reform_reward(self, reward, score, next_score, next_action_c, next_action,not_done):
        base_reward = -0.20* torch.tanh(score[:,2]-6)
        dynamic_reward = -0.125 * (next_score[: ,2]- score[:, 2])
        sofa_reward =  base_reward + dynamic_reward  
        sofa_reward = sofa_reward.unsqueeze(1)
        return sofa_reward


        
    def train(self, replay_buffer):
        # Sample replay buffer
        
        self.total_it += 1
        state, action_c, next_action_c, next_state, reward, scores,next_scores, outcome,  not_done = replay_buffer.sample()
        next_action = self.actor_target(next_state) 
    

        # Compute the target Q value
        # with torch.no_grad():
           
        next_action = self.actor(next_state) 
        #reward = outcome*torch.ones_like(reward)
        reward = self.reform_reward(reward, scores, next_scores, next_action_c, next_action_c, not_done)

        
        # Compute the target Q value
        target_Q1, target_Q2 = self.critic_target(next_state, next_action, next_scores[:,2].unsqueeze(1))
        target_Q = torch.min(target_Q1, target_Q2)
        
        #print(not_done)
        target_Q = reward + (not_done * self.discount * target_Q)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action_c, scores[:,2].unsqueeze(1))


        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 
       
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()
        
        
        
        self.writer.add_scalar('Loss/critic_loss', critic_loss,
                               self.total_it)

        self.writer.add_scalar('Loss/iv rl action', next_action[:,0].mean(),
                               self.total_it)

        self.writer.add_scalar('Loss/vc rl action', next_action[:,1].mean(),
                               self.total_it)

        self.writer.add_scalar('Loss/iv cl action', action_c[:,0].mean(),
                               self.total_it)

        self.writer.add_scalar('Loss/vc cl action', action_c[:,1].mean(),
                               self.total_it)


        


        


        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            #actor_loss = -self.critic.Q1(state, self.actor(state)).mean() + 5*  F.mse_loss(next_action_c, next_action) 
            cur_Q =   self.critic.Q1(state, self.actor(state), scores[:,2].unsqueeze(1)).mean()
            #sup_loss = custom_loss(action_c[:, 1], next_action[:,1])
            actor_loss = F.mse_loss(action_c, next_action)-0.2* cur_Q
            #actor_loss = sup_loss -0.2* cur_Q
            self.writer.add_scalar('Loss/Current Q value',cur_Q,
                               self.total_it)
            
            self.writer.add_scalar('Loss/actor_loss', actor_loss ,
                               self.total_it)
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return target_Q, current_Q1




def train_TD3(replay_buffer, test_replay_buffer, state_dim, action_dim,  device, parameters, writer):
    """
    one thing to note that the state dim is not the dim of raw data, it is the dimension output by auto-encoder
    which is hidden size used in experiments script
    """

    buffer_dir = parameters['buffer_dir']
    test_buffer_dir = parameters['test_buffer_dir']

    # Initialize and load policy
    policy = TD3(
        state_dim,
        action_dim,
        device,
        writer,
    )
    

    # Load replay buffer
    replay_buffer.load(buffer_dir, bootstrap=True)
    test_replay_buffer.load(test_buffer_dir, bootstrap=True)

  

    evaluations = []
    training_iters = 0

    # eval freq is essentially how often we write stuff to tensorboard
    while training_iters < parameters["max_timesteps"]:
        
        for _ in range(int(parameters["eval_freq"])):
            #policy.train(replay_buffer, training_iters)
            targ_q, cur_q = policy.train(replay_buffer)

        training_iters += int(parameters["eval_freq"])
        print(f"Training iterations: {training_iters}")

       # writer.add_scalar('Current Q value', torch.mean(targ_q), training_iters)

    direct_eval(policy, test_replay_buffer,  [0, 1], device, parameters)
    
    plot_action_dist(policy, test_replay_buffer,  parameters)
    plot_ucurve(policy, test_replay_buffer, device, parameters)
    plot_action_sofa(policy, test_replay_buffer,  parameters)
    
    



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
        state, action_c,next_action_c,  next_state, reward, scores, next_scores, outcome, done = replay_buffer.sample()
        
        action_rl = rl_policy.actor(state)
        batch_size = action_c.shape[0]
        iv_rand = torch.FloatTensor(batch_size, 1).uniform_(
            vc_range[0], vc_range[1])
        vc_rand = torch.FloatTensor(batch_size, 1).uniform_(
            vc_range[0], vc_range[1])
        action_rd = torch.cat((iv_rand ,vc_rand), axis=1).to(device)
        action_zero = torch.zeros_like(action_c).to(device)

        
        Q1_estimate, Q2_estimate = rl_policy.critic_target(state, action_rl, scores[:,2].unsqueeze(1))
        Q_estimate = torch.min(Q1_estimate, Q2_estimate).cpu()
        

        Q1_clinician, Q2_clinician = rl_policy.critic_target(state, action_c, scores[:,2].unsqueeze(1))
        Q_clinician = torch.min(Q1_clinician, Q2_clinician).cpu()


        Q1_random, Q2_random = rl_policy.critic_target(state, action_rd, scores[:,2].unsqueeze(1))
        Q_random = torch.min(Q1_random, Q2_random).cpu()

        Q1_zero, Q2_zero = rl_policy.critic_target(state, action_zero, scores[:,2].unsqueeze(1))
        Q_zero = torch.min(Q1_zero, Q2_zero).cpu()
        
        
        del action_rd  # release
        del action_zero
        
        Q_e += Q_estimate.sum()
       
        Q_c += Q_clinician.sum()
       
        Q_r += Q_random.sum()

        Q_z += Q_zero.sum()

        eval_iters += 1
    
    
    res_e = (Q_e/(eval_iters*batch_size)).item()
    res_c = (Q_c/(eval_iters*batch_size)).item()
    res_r = (Q_r/(eval_iters*batch_size)).item()
    res_z = (Q_z/(eval_iters*batch_size)).item()

    print('Q estimate', Q_e/(eval_iters*batch_size))
    print('Q clinician', Q_c/(eval_iters*batch_size))
    print('Q random', Q_r/(eval_iters*batch_size))
    print('Q zero', Q_z/(eval_iters*batch_size))

    import csv   
    fields=[res_e, res_c, res_r, res_z]
    with open('res.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

    
    
def plot_action_sofa(rl_policy, replay_buffer,  parameters):
    """
    """
    eval_iters = 0
    ivs_rl = []
    ivs_cl = []
    vcs_rl = []
    vcs_cl = []

    rl_policy.critic.eval()
    rl_policy.actor.eval()
    rl_policy.critic_target.eval()
    rl_policy.actor_target.eval()
    
    sofas = []
    while eval_iters < parameters["eval_steps"]:
        state, action_c, next_action_c, next_state, reward, scores,next_scores, outcome, done = replay_buffer.sample()
        action_rl =  rl_policy.actor(state)
        iv_rl = action_rl[:, 0]
        vc_rl = action_rl[:, 1]
        ivs_rl.append(iv_rl.detach().cpu().numpy())
        ivs_cl.append(action_c[:,0].cpu().numpy())
        vcs_rl.append(vc_rl.detach().cpu().numpy())
        vcs_cl.append(action_c[:,1].cpu().numpy())
        sofas.append(scores[:, 2].cpu().numpy())
        
        eval_iters+=1

    
    ivs_rl = np.concatenate(ivs_rl, axis=None)
    vcs_rl = np.concatenate(vcs_rl, axis=None)
    sofas = np.concatenate(sofas, axis=None)
    ivs_cl = np.concatenate(ivs_cl, axis=None)
    vcs_cl = np.concatenate(vcs_cl, axis=None)

    ivs_rl = ivs_rl * 2668
    ivs_cl = ivs_cl * 2668
    vcs_rl = vcs_rl * 1.187
    vcs_cl = vcs_cl * 1.187

    print('sofa vc', vcs_rl.mean() )
    print('sofa iv', ivs_rl.mean() )
    
    xs, vcrl_mean, vcrl_se, vccl_mean, vccl_se, ivcl_mean, ivcl_se, ivrl_mean, ivrl_se = [], [], [], [], [], [], [], [] ,[]
    for sofa in range(0, 25):
        sofa_idx = np.where(sofas==sofa)[0]
        xs.append(sofa)
        vccl_mean.append(vcs_cl[sofa_idx].mean())
        vccl_se.append(sem(vcs_cl[sofa_idx])) 
        vcrl_mean.append(vcs_rl[sofa_idx].mean())
        vcrl_se.append(sem(vcs_rl[sofa_idx])) 
        ivcl_mean.append(ivs_cl[sofa_idx].mean())
        ivcl_se.append(sem(ivs_cl[sofa_idx])) 
        ivrl_mean.append(ivs_rl[sofa_idx].mean())
        ivrl_se.append(sem(ivs_rl[sofa_idx])) 

    vccl_mean = np.array(vccl_mean)
    vccl_se = np.array(vccl_se)
    vcrl_mean = np.array(vcrl_mean)
    vcrl_se = np.array(vcrl_se)
    ivcl_mean = np.array(ivcl_mean)
    ivcl_se = np.array(ivcl_se)
    ivrl_mean = np.array(ivrl_mean)
    ivrl_se = np.array(ivrl_se)

    plt.figure()
    plt.subplot(121)
    plt.plot(xs,ivcl_mean, '-o', color='mediumseagreen', label = 'Clinician policy')
    plt.fill_between(xs, ivcl_mean - ivcl_se, ivcl_mean + ivcl_se, color='mediumseagreen', alpha=0.5)
    plt.plot(xs,ivrl_mean, '-o', color='darkgreen', label = 'RL policy')
    plt.fill_between(xs, ivrl_mean - ivrl_se, ivrl_mean + ivrl_se, color='darkgreen', alpha=0.5)
    plt.xlabel("SOFA score")
    plt.ylabel("Mean Dosage")
    plt.legend()
    plt.title('IV fluids')

    plt.subplot(122)
    plt.plot(xs,vccl_mean, '-o', color='skyblue', label = 'Clinician policy')
    plt.fill_between(xs, vccl_mean - vccl_se, vccl_mean + vccl_se, color='skyblue',alpha=0.5)
    plt.plot(xs,vcrl_mean, '-o', color='royalblue', label = 'RL policy')
    plt.fill_between(xs, vcrl_mean - vcrl_se, vcrl_mean + vcrl_se, color='royalblue', alpha=0.5)
    plt.xlabel("SOFA score")
    plt.ylabel("Mean Dosage")
    plt.legend()
    plt.title('Vasopressors')
    plt.show()


        
def plot_ucurve(rl_policy, replay_buffer, device, parameters):
    plt.figure()
    u_plot_iv(rl_policy, replay_buffer, device, parameters)
    u_plot_vc(rl_policy, replay_buffer, device, parameters)
    plt.show()


def plot_action_dist(rl_policy, replay_buffer,  parameters):
    f, (ax1, ax2) = plt.subplots(2, 1, sharey=False)
    ax1 = plt.subplot(2, 1, 1)
    
    eval_iters = 0
    
    xs_rl = []
    xs_cl = []
    ys_rl = []
    ys_cl = []

    rl_policy.critic.eval()
    rl_policy.actor.eval()
    rl_policy.critic_target.eval()
    rl_policy.actor_target.eval()
    while eval_iters < parameters["eval_steps"]:
        state, action_c, next_action_c, next_state, reward, scores,next_scores, outcome, done = replay_buffer.sample()
        action_rl =  rl_policy.actor_target(state)
        iv_rl = action_rl[:, 0]
        
        xs_rl.append(iv_rl.detach().cpu().numpy())
        xs_cl.append(action_c[:,0].cpu().numpy())

        vc_rl = action_rl[:, 1]
        ys_rl.append(vc_rl.detach().cpu().numpy())
        ys_cl.append(action_c[:,1].cpu().numpy())
        
        eval_iters+=1

    
    xs_rl = np.concatenate(xs_rl, axis=None)
    xs_cl = np.concatenate(xs_cl, axis=None)
    
    xs_cl = xs_cl * 2668
    xs_rl = xs_rl * 2668

    ys_rl = np.concatenate(ys_rl, axis=None)
    ys_rl = ys_rl * 1.187
    ys_cl = np.concatenate(ys_cl, axis=None)
    ys_cl = ys_cl * 1.187
    

    print('rl iv mean',xs_rl.mean())
    print('cl iv mean',xs_cl.mean())
    sns.set_style('whitegrid')
    ax = plt.subplot(2, 1, 1)
    sns.distplot(xs_cl, color='mediumseagreen', kde_kws={"clip":(0,2000)}, hist_kws={"range":(0,2000)}, ax = ax1)
    sns.distplot(xs_rl, color='darkgreen', kde_kws={"clip":(0,2000)}, hist_kws={"range":(0,2000)}, ax = ax1)

    ax.title.set_text('IV fluids dosage distribution')
    ax.legend(['Clinician policy', 'RL policy'])
    ax2 = plt.subplot(2, 1, 2)
    ax2.set_yscale('log')

    sns.distplot(ys_cl, color='skyblue', kde_kws={"clip":(0.0, 1.0)}, hist_kws={"range":(0.0,1.0)},ax = ax2)
    sns.distplot(ys_rl, color='royalblue', kde_kws={"clip":(0.0, 1.0)}, hist_kws={"range":(0.0,1.0)}, ax = ax2)
    
    ax2.title.set_text('Vasopressors dosage distribution')
    ax2.legend(['Clinician policy', 'RL policy'])
    
    plt.show()


def u_plot_vc(rl_policy, replay_buffer, device, parameters):
    plt.subplot(122)
    from scipy.stats import sem
    eval_iters = 0
    vc_diffs_rl = []
    vc_diffs_no = []
    death =[]
    while eval_iters < parameters["eval_steps"]:
        state, action_c, next_action_c, next_state, reward, scores, next_scores, outcome, done = replay_buffer.sample()
        no_action = torch.zeros_like(next_action_c)
        #action_c = action_c[:,1].reshape(-1, 1)
        outcome = torch.where(outcome == -1, 1, 0)
        # print(state.sum())
        action_rl = rl_policy.actor(state)
        action_rl = action_rl * 1.187
        next_action_c = next_action_c * 1.187
        batch_size = action_c.shape[0]
        
        vc_r = action_rl[:,1] - next_action_c[:,1]
        vc_n = no_action[:, 1] - next_action_c[:, 1]
        #vc_d = action_rl - action_c
        
        vc_diffs_rl.append(vc_r.cpu().detach().numpy())
        vc_diffs_no.append(vc_n.cpu().detach().numpy())

        death.append(outcome.cpu().detach().numpy())
        eval_iters+=1
    
    print('data ready')
    
    vc_diffs_rl = np.concatenate(vc_diffs_rl, axis = None)
    vc_diffs_no = np.concatenate(vc_diffs_no, axis = None)

    
    death = np.concatenate(death, axis = None)
    
        

    mort_vc_rl = []
    mort_vc_no = []
    bin_vc= []
    std_vc_rl = []
    std_vc_no = []
    i = 1.0
    while i >=-0.6:
        idx_vc_rl = np.where((vc_diffs_rl>i-0.05) & (vc_diffs_rl<i+0.05))[0]
        death_vc_rl = death[idx_vc_rl]

        idx_vc_no = np.where((vc_diffs_no>i-0.05) & (vc_diffs_no<i+0.05))[0]
        death_vc_no = death[idx_vc_no]

        
        death_mean_vc_rl = (death_vc_rl.sum())/len(death_vc_rl)
        death_mean_vc_no = (death_vc_no.sum())/len(death_vc_no)

        
        death_se_vc_rl = sem(death_vc_rl)
        death_se_vc_no = sem(death_vc_no)


        mort_vc_rl.append(death_mean_vc_rl)
        mort_vc_no.append(death_mean_vc_no)


        bin_vc.append(i)
        std_vc_rl.append(death_se_vc_rl)
        std_vc_no.append(death_se_vc_no)
        i-=0.1

    mort_vc_rl  = np.array(mort_vc_rl)
    mort_vc_no  = np.array(mort_vc_no)
    bin_vc = np.array(bin_vc)
    std_vc_rl = np.array(std_vc_rl)
    std_vc_no = np.array(std_vc_no)

    #fig, axes = plt.subplots(1, 2)

    plt.plot(bin_vc, mort_vc_rl, color='skyblue', label = 'RL policy')
    plt.fill_between(bin_vc, mort_vc_rl - 1*std_vc_rl,  mort_vc_rl + 1*std_vc_rl, color='skyblue', alpha = 0.5)
    

    plt.plot(bin_vc, mort_vc_no, color='royalblue', label = 'Clinician policy')
    plt.fill_between(bin_vc, mort_vc_no - 1*std_vc_no,  mort_vc_no + 1*std_vc_no, color='royalblue', alpha = 0.5)

    plt.xlabel("Recommend minus clinician dosage")
    plt.ylabel("Observed mortality")
    plt.title('Vasopressors')

    plt.legend()
    


def u_plot_iv(rl_policy, replay_buffer, device, parameters):
    plt.subplot(121)
    from scipy.stats import sem
    eval_iters = 0
    iv_diffs_rl, iv_diffs_no = [], []
    death =[]
    while eval_iters < parameters["eval_steps"]:
        state, action_c, next_action_c, next_state, reward, scores, next_scores, outcome, done = replay_buffer.sample()
        no_action = torch.zeros_like(next_action_c)
        outcome = torch.where(outcome == -1, 1, 0)
        action_rl = rl_policy.actor(state) 
        action_rl = action_rl * 2668
        next_action_c = next_action_c * 2668

        batch_size = action_c.shape[0]
        iv_r = action_rl[:,0] - next_action_c[:,0]
        iv_n = no_action[:, 0] - next_action_c[:, 0]
        
        #vc_d = action_rl - action_c
        iv_diffs_rl.append(iv_r.cpu().detach().numpy())
        iv_diffs_no.append(iv_n.cpu().detach().numpy())
       

        death.append(outcome.cpu().detach().numpy())
        eval_iters+=1
    
    print('data ready')
    iv_diffs_rl = np.concatenate(iv_diffs_rl, axis=None)
    iv_diffs_no = np.concatenate(iv_diffs_no, axis=None)

    death = np.concatenate(death, axis = None)
    

    mort_iv_no, mort_iv_rl = [],[]
    bin_iv = []
    std_iv_rl = []
    std_iv_no =  []
    i = 1200
    while i >=-1200:
        idx_iv_rl = np.where((iv_diffs_rl>i-100) & (iv_diffs_rl<i+100))[0]
        death_iv_rl = death[idx_iv_rl]

        idx_iv_no = np.where((iv_diffs_no>i-100) & (iv_diffs_no<i+100))[0]
        death_iv_no = death[idx_iv_no]
        
        death_mean_iv_rl = (death_iv_rl.sum())/len(death_iv_rl)
        death_mean_iv_no = (death_iv_no.sum())/len(death_iv_no)
        
        death_se_iv_rl = sem(death_iv_rl)
        death_se_iv_no = sem(death_iv_no)

        mort_iv_rl.append(death_mean_iv_rl)
        mort_iv_no.append(death_mean_iv_no)

        bin_iv.append(i)
        std_iv_rl.append(death_se_iv_rl)

        std_iv_no.append(death_se_iv_no)
        i-=100

    mort_iv_rl  = np.array(mort_iv_rl)
    mort_iv_no  = np.array(mort_iv_no)

    bin_iv = np.array(bin_iv)
    std_iv_rl = np.array(std_iv_rl)
    std_iv_no = np.array(std_iv_no)

    #fig, axes = plt.subplots(1, 2)

    plt.plot(bin_iv, mort_iv_rl, color='mediumseagreen', label = 'RL policy')
    plt.fill_between(bin_iv, mort_iv_rl - 1*std_iv_rl,  mort_iv_rl + 1*std_iv_rl, color='mediumseagreen', alpha = 0.5)
    

    plt.plot(bin_iv, mort_iv_no, color='darkgreen', label = 'Clinician policy')
    plt.fill_between(bin_iv, mort_iv_no - 1*std_iv_no,  mort_iv_no + 1*std_iv_no, color='darkgreen', alpha = 0.5)

    plt.xlabel("Recommend minus clinician dosage")
    plt.ylabel("Observed mortality")
    plt.title('IV fluids')

    plt.legend()
    

   
    




    


    
