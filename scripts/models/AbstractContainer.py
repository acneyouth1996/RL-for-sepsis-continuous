from abc import ABC,abstractmethod 
import torch
from .common import pearson_correlation

class AbstractContainer(ABC): 
    '''
    Abstract class for a model container.
    Contains self.gen (defined after calling make_encoder)
             self.pred (defined after calling make_decoder)
    The loop function runs one batch of inputs through the encoder and decoder, and returns the loss.
    Models other than [AE, AIS, RNN] should overload the loop function.
    '''
    @abstractmethod
    def __init__(self, device, **kwargs): 
        pass
    
    @abstractmethod
    def make_encoder(self, **kwargs): 
        pass
    
    @abstractmethod
    def make_decoder(self, **kwargs): 
        pass
    
    def loop(self, obs, dem,  actions_c,  device='cpu', continous=True, **kwargs):
        '''This loop through the training and validation data is the general template for AIS, RNN, etc'''
        # Split the observations 
        autoencoder = kwargs['autoencoder']
        cur_obs, next_obs = obs[:,:-1,:], obs[:,1:,:]
        cur_dem = dem[:,:-1,:]
        # cur_scores, next_scores = scores[:,:-1,:], scores[:,1:,:] # I won't need the "next scores"
        mask = (cur_obs ==0).all(dim=2) # Compute mask for extra appended rows of observations (all zeros along dim 2)

        # This concatenates an empty action with the first observation and shifts all actions 
        # to the next observation since we're interested in pairing obs with previous action
        
        hidden_states = self.gen(torch.cat((cur_obs, cur_dem, torch.cat((torch.zeros((obs.shape[0],1,actions_c.shape[-1])).to(device),actions_c[:,:-2,:]),dim=1)),dim=-1))

        if autoencoder == 'RNN':
            pred_obs = self.pred(hidden_states)
        else:
            pred_obs = self.pred(torch.cat((hidden_states, actions_c[:,:-1,:]),dim=-1))
        
        
        temp_loss = -torch.distributions.MultivariateNormal(pred_obs, torch.eye(pred_obs.shape[-1]).to(device)).log_prob(next_obs)
        mse_loss = sum(temp_loss[~mask])
        
        return  mse_loss, hidden_states