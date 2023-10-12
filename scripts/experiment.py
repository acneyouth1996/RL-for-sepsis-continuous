'''
This module defines the Experiment class that intializes, trains, and evaluates a Recurrent autoencoder.

The central focus of this class is to develop representations of sequential patient states in acute clinical settings.
These representations are learned through an auxiliary task of predicting the subsequent physiological observation but 
are also used to train a treatment policy via offline RL. The specific policy learning algorithm implemented through this
module is the discretized form of Batch Constrained Q-learning [Fujimoto, et al (2019)]

This module was designed and tested for use with a Septic patient cohort extracted from the MIMIC-III (v1.4) database. It is
assumed that the data used to create the Dataloaders in lines 174, 180 and 186 is patient and time aligned separate sequences 
of:
    (1) patient demographics
    (2) observations of patient vitals, labs and other relevant tests
    (3) assigned treatments or interventions
    (4) how long each patient trajectory is
    (5) corresponding patient acuity scores, and
    (6) patient outcomes (here, binary - death vs. survival)

The cohort used and evaluated in the study this code was built for is defined at: https://github.com/microsoft/mimic_sepsis
============================================================================================================================
This code is provided under the MIT License and is meant to be helpful, but WITHOUT ANY WARRANTY;

November 2020 by Taylor Killian and Haoran Zhang; University of Toronto + Vector Institute
Modified by Yong Huang 2022 at UCI
============================================================================================================================
Notes:
 - The code for the AIS approach and general framework we build from was developed by Jayakumar Subramanian

'''
import numpy as np
import pandas as pd
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from itertools import chain
from utils import ReplayBuffer
import os
import copy
import pickle
import ddpg_utils
import td3_utils
from models import AE,RNN



class Experiment(object): 
    def __init__(self, domain, train_data_file, validation_data_file, test_data_file, writer, minibatch_size, device,
                context_dim=5, folder_name='./data', autoencoder_saving_period=20,
                autoencoder_num_epochs=50, autoencoder_lr=0.001, autoencoder='AE', hidden_size=16, 
                state_dim=42, action_dim=2, score_dim = 3, rl_method= 'ddpg', **kwargs):
        '''
        We assume continuous actions and scalar rewards!
        '''
        self.device = device
        self.train_data_file = train_data_file
        self.validation_data_file = validation_data_file
        self.test_data_file = test_data_file
        self.minibatch_size = minibatch_size
        self.autoencoder_num_epochs = autoencoder_num_epochs 
        self.autoencoder = autoencoder
        self.autoencoder_lr = autoencoder_lr
        self.saving_period = autoencoder_saving_period
        self.state_dim = state_dim
        self.score_dim = score_dim
        self.action_dim = action_dim
        self.writer = writer

        self.context_dim = context_dim # Check to see if we'll remove the context from the input and only use it for decoding
        self.hidden_size = hidden_size
        
       
        self.input_dim = self.state_dim + self.context_dim + self.action_dim
        
        
        self.autoencoder_lower = self.autoencoder.lower()
        self.data_folder = folder_name + f'/{self.autoencoder_lower}_data'
        self.checkpoint_file = folder_name + f'/{self.autoencoder_lower}_checkpoints/checkpoint.pt'
        if not os.path.exists(folder_name + f'/{self.autoencoder_lower}_checkpoints'):
            os.mkdir(folder_name + f'/{self.autoencoder_lower}_checkpoints')
        if not os.path.exists(folder_name + f'/{self.autoencoder_lower}_data'):
            os.mkdir(folder_name + f'/{self.autoencoder_lower}_data')
        self.gen_file = folder_name + f'/{self.autoencoder_lower}_data/{self.autoencoder_lower}_gen.pt'
        self.pred_file = folder_name + f'/{self.autoencoder_lower}_data/{self.autoencoder_lower}_pred.pt'
        
        if self.autoencoder == 'AE':
            self.container = AE.ModelContainer(device)
            self.gen = self.container.make_encoder(self.hidden_size, self.state_dim, self.action_dim, context_dim=self.context_dim)
            self.pred = self.container.make_decoder(self.hidden_size, self.state_dim, self.action_dim)
        
        elif self.autoencoder == 'RNN':
            self.container = RNN.ModelContainer(device)
            
            self.gen = self.container.make_encoder(self.hidden_size, self.state_dim, self.action_dim,  context_dim=self.context_dim)
            self.pred = self.container.make_decoder(self.hidden_size, self.state_dim, self.action_dim)


        self.rl_method = rl_method

        self.buffer_save_file = self.data_folder + '/ReplayBuffer'
        self.test_buffer_save_file = self.data_folder + '/TestReplayBuffer'
        self.policy_save_file = self.data_folder + '/{}_policy'.format(rl_method)

        
        # Read in the data csv files
        assert (domain=='sepsis')        
        self.train_demog, self.train_states, self.train_interventions_c, self.train_lengths, self.train_times, self.acuities,\
             self.rewards, self.outcomes = torch.load(self.train_data_file)
        train_idx = torch.arange(self.train_demog.shape[0])
        print('before loading', torch.count_nonzero(self.train_interventions_c))
        self.train_dataset = TensorDataset(self.train_demog, self.train_states, self.train_interventions_c, self.train_lengths,\
            self.train_times, self.acuities, self.rewards, self.outcomes,  train_idx)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.minibatch_size, shuffle=True)

        self.val_demog, self.val_states,  self.val_interventions_c, self.val_lengths, self.val_times, self.val_acuities,\
             self.val_rewards, self.val_outcomes = torch.load(self.validation_data_file)
        val_idx = torch.arange(self.val_demog.shape[0])
        self.val_dataset = TensorDataset(self.val_demog, self.val_states, self.val_interventions_c, self.val_lengths,\
             self.val_times, self.val_acuities, self.val_rewards,self.val_outcomes, val_idx)

        self.val_loader = DataLoader(self.val_dataset, batch_size=self.minibatch_size, shuffle=False)

        self.test_demog, self.test_states,  self.test_interventions_c, self.test_lengths, self.test_times, self.test_acuities, \
            self.test_rewards, self.test_outcomes = torch.load(self.test_data_file)
        test_idx = torch.arange(self.test_demog.shape[0])
        self.test_dataset = TensorDataset(self.test_demog, self.test_states,  self.test_interventions_c, self.test_lengths,\
             self.test_times, self.test_acuities, self.test_rewards, self.test_outcomes, test_idx)

        self.test_loader = DataLoader(self.test_dataset, batch_size=self.minibatch_size, shuffle=False)
        
    
    def load_model_from_checkpoint(self, checkpoint_file_path):
        checkpoint = torch.load(checkpoint_file_path)
        self.gen.load_state_dict(checkpoint['{}_gen_state_dict'.format(self.autoencoder.lower())])
        self.pred.load_state_dict(checkpoint['{}_pred_state_dict'.format(self.autoencoder.lower())])
        print("Experiment: generator and predictor models loaded.")

    def train_autoencoder(self):
        print(self.autoencoder)
        if self.autoencoder == 'None':
            return
        train_batch_counter = 0
        print('Experiment: training autoencoder')
        device = self.device
        self.optimizer = torch.optim.Adam(list(self.gen.parameters()) + list(self.pred.parameters()), lr=self.autoencoder_lr, amsgrad=True)

        self.autoencoding_losses = []
        self.autoencoding_losses_validation = []
        epoch_0 = 0

        for epoch in range(epoch_0, self.autoencoder_num_epochs):
            epoch_loss = []
            print("Experiment: autoencoder {0}: training Epoch = ".format(self.autoencoder), epoch+1, 'out of', self.autoencoder_num_epochs, 'epochs')

            # Loop through the data using the data loader
            for ii, (dem, ob, acc, l, t, scores, rewards, outcomes,  idx) in enumerate(self.train_loader):
                train_batch_counter += 1
                # print("Batch {}".format(ii),end='')
                dem = dem.to(device)  # 5 dimensional vector (Gender, Ventilation status, Re-admission status, Age, Weight)
                ob = ob.to(device)    # 33 dimensional vector (time varying measures)
                acc = acc.to(device)
                l = l.to(device)
                t = t.to(device)
                scores = scores.to(device)
                idx = idx.to(device)

                # Cut tensors down to the batch's largest sequence length... Trying to speed things up a bit...
                max_length = int(l.max().item())               
                    
                self.gen.train()
                self.pred.train()
                
                ob = ob[:,:max_length,:]
                dem = dem[:,:max_length,:]
                acc = acc[:,:max_length,:]
                scores = scores[:,:max_length,:]
                
                
                mse_loss, _ = self.container.loop(ob, dem, acc,  device=device, autoencoder = self.autoencoder)   

                self.optimizer.zero_grad()
                
                
                mse_loss.backward()
                self.optimizer.step()
                epoch_loss.append(mse_loss.detach().cpu().numpy())
                self.writer.add_scalar('Autoencoder training loss', mse_loss.detach().cpu().numpy(), train_batch_counter)                
                
                                        
            self.autoencoding_losses.append(epoch_loss)
            if (epoch+1)%self.saving_period == 0: # Run validation and also save checkpoint
                
                #Computing validation loss
                epoch_validation_loss = []
                with torch.no_grad():
                    for jj, (dem, ob,acc, l, t, scores, rewards, outcomes, idx) in enumerate(self.val_loader):
                        dem = dem.to(device)
                        ob = ob.to(device)
                        acc = acc.to(device)
                        l = l.to(device)
                        t = t.to(device)
                        idx = idx.to(device)
                        

                        # Cut tensors down to the batch's largest sequence length... Trying to speed things up a bit...
                        max_length = int(l.max().item())                        
                        
                        ob = ob[:,:max_length,:]
                        dem = dem[:,:max_length,:]
                        acc = acc[:,:max_length,:]
                        
                        scores = scores[:,:max_length,:] 
                        
                        self.gen.eval()
                        self.pred.eval()    
            
                        mse_loss, _ = self.container.loop(ob, dem, acc, device=device, autoencoder = self.autoencoder)                                                 
                        epoch_validation_loss.append(mse_loss.detach().cpu().numpy())
                    
                        
                self.autoencoding_losses_validation.append(epoch_validation_loss)

                save_dict = {'epoch': epoch,
                        'gen_state_dict': self.gen.state_dict(),
                        'pred_state_dict': self.pred.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': self.autoencoding_losses,
                        'validation_loss': self.autoencoding_losses_validation
                        }
                
                    
                try:
                    torch.save(save_dict, self.checkpoint_file)
                    np.save(self.data_folder + '/{}_losses.npy'.format(self.autoencoder.lower()), np.array(self.autoencoding_losses))
                except Exception as e:
                    print(e)
                
                try:
                    np.save(self.data_folder + '/{}_validation_losses.npy'.format(self.autoencoder.lower()), np.array(self.autoencoding_losses_validation))
                except Exception as e:
                    print(e)
                    
            #Final epoch checkpoint
            try:
                save_dict = {
                            'epoch': self.autoencoder_num_epochs-1,
                            'gen_state_dict': self.gen.state_dict(),
                            'pred_state_dict': self.pred.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': self.autoencoding_losses,
                            'validation_loss': self.autoencoding_losses_validation,
                            }
                torch.save(self.gen.state_dict(), self.gen_file)
                torch.save(self.pred.state_dict(), self.pred_file)
                torch.save(save_dict, self.checkpoint_file)
                np.save(self.data_folder + '/{}_losses.npy'.format(self.autoencoder.lower()), np.array(self.autoencoding_losses))
            except Exception as e:
                    print(e)
           
        
    def evaluate_trained_model(self):
        '''After training, this method can be called to use the trained autoencoder to embed all the data in the representation space.
        We encode all data subsets (train, validation and test) separately and save them off as independent tuples. We then will
        also combine these subsets to populate a replay buffer to train a policy from.
        
        '''
        # Initialize the replay buffer
        if self.autoencoder != 'None':
            self.replay_buffer = ReplayBuffer(self.hidden_size, self.minibatch_size, 350000, self.device, True)
            self.test_replay_buffer = ReplayBuffer(self.hidden_size, self.minibatch_size, 350000, self.device, True)
        else:
            self.replay_buffer = ReplayBuffer(self.state_dim + self.context_dim , self.minibatch_size, 350000, self.device, False)
            self.test_replay_buffer = ReplayBuffer(self.state_dim + self.context_dim , self.minibatch_size, 350000, self.device, False)

        
        print('Encoding the Training and Validataion Data.')
        ## LOOP THROUGH THE DATA
        # -----------------------------------------------
        # For Training and Validation sets (Encode the observations only, add all data to the experience replay buffer)
        # For the Test set:
        # - Encode the observations
        # - Save off the data (as test tuples and place in the experience replay buffer)
        # - Evaluate accuracy of predicting the next observation using the decoder module of the model
        splits = ['train', 'val', 'test']
        with torch.no_grad():
            for i_set, loader in enumerate([self.train_loader, self.val_loader, self.test_loader]):
                counter = 0
                if i_set == 2:
                    print('Encoding the Test Data. Evaluating prediction errors.')
                for dem, ob, acc, l, t, scores, rewards, outcomes, idx in loader:
                    counter += 1
                    dem = dem.to(self.device)
                    ob = ob.to(self.device)
                    
                    acc = acc.to(self.device)
                    l = l.to(self.device)
                    t = t.to(self.device)
                    rewards = rewards.to(self.device)

                    max_length = int(l.max().item())

                    ob = ob[:,:max_length,:]
                    dem = dem[:,:max_length,:]
                    acc = acc[:,:max_length,:]
                    scores = scores[:,:max_length,:]
                    rewards = rewards[:,:max_length]
                    outcomes = outcomes[:,:max_length]

                    cur_obs, next_obs = ob[:,:-1,:], ob[:,1:,:]
                    cur_dem, next_dem = dem[:,:-1,:], dem[:,1:,:]
                    cur_scores = scores[:, :-1, :]
                    next_scores = scores[:, 1:, :]
                    

                    cur_actions_c = acc[:,:-1,:]
                    next_actions_c = acc[:, 1:, :]
                    cur_rewards = rewards[:,:-1]
                    outcomes = outcomes[:,:-1]

                    mask = (cur_obs==0).all(dim=2)
                    
                    
                    
                    if self.autoencoder != 'None':
                        
                        self.gen.eval()
                        self.pred.eval()

                        representations = self.gen(torch.cat((cur_obs, cur_dem, torch.cat((torch.zeros((ob.shape[0],1,acc.shape[-1])).to(self.device),acc[:,:-2,:]),dim=1)),dim=-1))
                        if self.autoencoder == 'RNN':
                            pred_obs = self.pred(representations)
                        else:
                            pred_obs = self.pred(torch.cat((representations, cur_actions_c),dim=-1))


                        pred_error = F.mse_loss(next_obs[~mask], pred_obs[~mask])
                        self.writer.add_scalar('Autoencoder pred errors on {} set'.format(splits[i_set]),\
                            pred_error.detach().cpu().numpy(), counter)
                                                

                        # Remove values with the computed mask and add data to the experience replay buffer
                        cur_rep = torch.cat((representations[:,:-1, :], torch.zeros((cur_obs.shape[0], 1, self.hidden_size)).to(self.device)), dim=1)
                        next_rep = torch.cat((representations[:,1:, :], torch.zeros((cur_obs.shape[0], 1, self.hidden_size)).to(self.device)), dim=1)
                        cur_rep = cur_rep[~mask].cpu()
                        next_rep = next_rep[~mask].cpu()

                        cur_actions_c = cur_actions_c[~mask].cpu()
                        next_actions_c = next_actions_c[~mask].cpu()
                        cur_scores = cur_scores[~mask].cpu()
                        next_scores = next_scores[~mask].cpu()
                        cur_rewards = cur_rewards[~mask].cpu()
                        outcomes = outcomes[~mask].cpu()
                        
                        # Loop over all transitions and add them to the replay buffer
                        for i_trans in range(cur_rep.shape[0]):
                            done = cur_rewards[i_trans] != 0
                            if i_set != 2:
                                #print(cur_actions_c[i_trans].shape)
                                #print(cur_scores[i_trans].shape)
                                self.replay_buffer.add(cur_rep[i_trans].numpy(), cur_actions_c[i_trans].numpy(), next_actions_c[i_trans].numpy(), next_rep[i_trans].numpy(), cur_rewards[i_trans].item(), cur_scores[i_trans].numpy(), next_scores[i_trans].numpy(), outcomes[i_trans].numpy(), done.item())

                            else:
                                self.test_replay_buffer.add(cur_rep[i_trans].numpy(), cur_actions_c[i_trans].numpy(),next_actions_c[i_trans].numpy(), next_rep[i_trans].numpy(), cur_rewards[i_trans].item(),cur_scores[i_trans].numpy(), next_scores[i_trans].numpy(), outcomes[i_trans].numpy(), done.item())
                    else:
                        
                        cur_scores = cur_scores[~mask].cpu()
                        next_scores = next_scores[~mask].cpu()
                        cur_obs = cur_obs[~mask].cpu()
                        next_obs = next_obs[~mask].cpu()
                        cur_dem = cur_dem[~mask].cpu()
                        next_dem = next_dem[~mask].cpu()
                        cur_actions_c = cur_actions_c[~mask].cpu()
                        next_actions_c = next_actions_c[~mask].cpu()
                        cur_rewards = cur_rewards[~mask].cpu()
                        outcomes = outcomes[~mask].cpu()
                        
                        
                        for i_trans in range(cur_rewards.shape[0]):

                            
                            done = cur_rewards[i_trans] != 0
                            
                            
                            
                            if i_set!= 2:
                                self.replay_buffer.add(torch.cat((cur_obs[i_trans],cur_dem[i_trans]),dim=-1).numpy(), cur_actions_c[i_trans].numpy(), next_actions_c[i_trans].numpy(), torch.cat((next_obs[i_trans], next_dem[i_trans]), dim=-1).numpy(),\
                                    cur_rewards[i_trans].item(), cur_scores[i_trans].numpy(), next_scores[i_trans].numpy(), outcomes[i_trans].numpy(), done.item())
                            else:
                                self.test_replay_buffer.add(torch.cat((cur_obs[i_trans],cur_dem[i_trans]),dim=-1).numpy(), cur_actions_c[i_trans].numpy(), next_actions_c[i_trans].numpy(), torch.cat((next_obs[i_trans], next_dem[i_trans]), dim=-1).numpy(),\
                                    cur_rewards[i_trans].item(), cur_scores[i_trans].numpy(), next_scores[i_trans].numpy(),outcomes[i_trans].numpy(), done.item())



            ## SAVE OFF DATA
            # maybe need to save train_representation and val_representation file as well
            # --------------
            self.replay_buffer.save(self.buffer_save_file)
            self.test_replay_buffer.save(self.test_buffer_save_file)

           
            


    def train_DDPG_policy(self, actor_learning_rate, critic_learning_rate):

        # Initialize parameters for policy learning
        params = {
            "eval_freq": 500,
            'eval_steps': 5000,
            "discount": 0.99,
            "buffer_size": 350000,
            "batch_size": self.minibatch_size,
            "optimizer": "Adam",
            "actor_optimizer_parameters": {
                "lr": actor_learning_rate
            },
            "critic_optimizer_parameters":{
                "lr": critic_learning_rate
            },
            "train_freq": 1,
            "polyak_target_update": True,
            "target_update_freq": 2,
            "tau": 0.01,
            "max_timesteps": 3e2,
            "buffer_dir": self.buffer_save_file,
            "test_buffer_dir": self.test_buffer_save_file,
            "policy_file": self.policy_save_file+f'_l{actor_learning_rate}'+f'_l{critic_learning_rate}' +'.pt',
            
        }
        
        #pol_eval_dataset = TensorDataset(test_representations, self.test_states, self.test_interventions_c, self.test_demog, self.test_rewards)
        

        # Initialize and Load the experience replay buffer corresponding with the current settings of rand_num, hidden_size, etc...
        if self.autoencoder != 'None':
            replay_buffer = ReplayBuffer(self.hidden_size, self.minibatch_size, 350000, self.device, True)
            test_replay_buffer = ReplayBuffer(self.hidden_size, self.minibatch_size, 350000, self.device, True)
            state_dim = self.hidden_size
        else:
            replay_buffer = ReplayBuffer(self.state_dim + self.context_dim, self.minibatch_size, 350000, self.device, False)
            test_replay_buffer = ReplayBuffer(self.state_dim + self.context_dim , self.minibatch_size, 350000, self.device, False)
            state_dim = self.state_dim + self.context_dim

        
        

        # Run dBCQ_utils.train_dBCQ
        max_action = torch.FloatTensor([1.0, 1.0]).to(self.device)
#
        ddpg_utils.train_DDPG(replay_buffer, test_replay_buffer, state_dim, self.action_dim, self.device, params, self.writer)
        #ddpg_utils.train_BC(replay_buffer, test_replay_buffer, state_dim, self.action_dim, max_action, self.device, params, self.writer)

    def train_TD3_policy(self, actor_learning_rate, critic_learning_rate):

        # Initialize parameters for policy learning
        params = {
            "eval_freq": 500,
            'eval_steps': 10000,
            "discount": 0.99,
            "buffer_size": 350000,
            "batch_size": self.minibatch_size,
            "optimizer": "Adam",
            
            
            "tau": 0.01,
            "max_timesteps": 2e4,
            "buffer_dir": self.buffer_save_file,
            "test_buffer_dir": self.test_buffer_save_file,
            "policy_file": self.policy_save_file+f'_l{actor_learning_rate}'+f'_l{critic_learning_rate}' +'.pt',
            
        }
        
        #pol_eval_dataset = TensorDataset(test_representations, self.test_states, self.test_interventions_c, self.test_demog, self.test_rewards)
        

        # Initialize and Load the experience replay buffer corresponding with the current settings of rand_num, hidden_size, etc...
        if self.autoencoder != 'None':
            replay_buffer = ReplayBuffer(self.hidden_size, self.minibatch_size, 350000, self.device, True)
            test_replay_buffer = ReplayBuffer(self.hidden_size, self.minibatch_size, 350000, self.device, True)
            state_dim = self.hidden_size
        else:
            replay_buffer = ReplayBuffer(self.state_dim + self.context_dim, self.minibatch_size, 350000, self.device, False)
            test_replay_buffer = ReplayBuffer(self.state_dim + self.context_dim , self.minibatch_size, 350000, self.device, False)
            state_dim = self.state_dim + self.context_dim

        
        

        # Run dBCQ_utils.train_dBCQ
        max_action = torch.FloatTensor([1.0, 1.0]).to(self.device)
#
        td3_utils.train_TD3(replay_buffer, test_replay_buffer, state_dim, self.action_dim, self.device, params, self.writer)
        