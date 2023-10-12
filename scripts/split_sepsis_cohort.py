import numpy as np
import pandas as pd
import os
import torch
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn import preprocessing

ts = pd.read_csv('../data/sepsis_final_data_withTimes.csv')
raw_ts = pd.read_csv('../data/sepsis_final_data_RAW_withTimes.csv')

ts_idx = ts[(ts['a_vc:action_vc'] < ts['a_vc:action_vc'].quantile(0.990)) & (ts['a_iv:action_iv'] < ts['a_iv:action_iv'].quantile(0.990))].index
ts = ts.iloc[ts_idx]
raw_ts = raw_ts.iloc[ts_idx]
acuity_scores = pd.read_csv('../acuity_scores.csv')
acuity_scores = acuity_scores.iloc[ts_idx]
acuity_scores['o:SOFA'] = raw_ts['o:SOFA']

def normalize(arr):
    min_v = arr.min()
    max_v = arr.max()
    arr = (arr-min_v)/(max_v-min_v)
    return arr

ts['a_iv:action_iv'] = normalize(ts['a_iv:action_iv']) 
ts['a_vc:action_vc'] = normalize(ts['a_vc:action_vc'])

ts['outcome'] = 0.0
for i in ts['traj'].unique().tolist():
    outcome = ts[ts['traj']==i]['r:reward'].sum()
    ts.loc[ts['traj']==i, 'outcome'] = outcome

## Determine the train, val, test split (70/15/15), stratified by patient outcome
temp = ts.groupby('traj')['r:reward'].sum()


y = temp.values
X = temp.index.values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, stratify=y_test, test_size=0.5)

# Drop unneeded meta features
full_zs = ts.drop(['m:presumed_onset', 'm:charttime', 'm:icustayid'], axis=1)

train_data = full_zs[full_zs['traj'].isin(X_train)]
train_acuity = acuity_scores[acuity_scores['traj'].isin(X_train)]
trajectories = train_data['traj'].unique()

# Define the features of the full data
num_c_actions = 2
state_dim = 47
num_obs = 33
num_dem = 5
num_acuity_scores = 3
# horizon should be the max length of a traj we want to keep
horizon = 21
device = 'cpu'

dem_keep_cols = ['o:gender', 'o:mechvent', 'o:re_admission', 'o:age', 'o:Weight_kg']
obs_keep_cols = ['o:GCS', 'o:HR', 'o:SysBP',
       'o:MeanBP', 'o:DiaBP', 'o:RR', 'o:Temp_C', 'o:FiO2_1', 'o:Potassium',
       'o:Sodium', 'o:Chloride', 'o:Glucose', 'o:Magnesium', 'o:Calcium',
       'o:Hb', 'o:WBC_count', 'o:Platelets_count', 'o:PTT', 'o:PT',
       'o:Arterial_pH', 'o:paO2', 'o:paCO2', 'o:Arterial_BE', 'o:HCO3',
       'o:Arterial_lactate','o:PaO2_FiO2', 'o:SpO2', 'o:BUN', 'o:Creatinine',
       'o:SGOT', 'o:SGPT', 'o:Total_bili', 'o:INR']

dem_cols = [i for i in train_data.columns if i in dem_keep_cols]
obs_cols = [i for i in train_data.columns if i in obs_keep_cols]
acc_cols  = [i for i in train_data.columns if i[:2] == 'a_']
rew_cols = [i for i in train_data.columns if i[:2] == 'r:']
acuity_cols = [i for i in train_acuity.columns if i in ['c:OASIS','c:SAPSii','o:SOFA']]
outcome_cols = [i for i in train_data.columns if i == 'outcome']

assert len(obs_cols) > 0, 'No observations present, or observation columns not prefixed with "o:"'
assert len(rew_cols) > 0, 'No rewards present, or rewards column not prefixed with "r:"'
assert len(acc_cols) == 2, 'More than 2 action columns are present when 2 action column is expected'
assert len(rew_cols) == 1, 'Multiple reward columns are present when a single reward column is expected'
assert len(acuity_cols) == num_acuity_scores, 'Ensure that we have the right number of acuity scores'


rew_col = rew_cols[0]
outcome_col = outcome_cols[0]

data_trajectory = {}
data_trajectory['dem_cols'] = dem_cols
data_trajectory['obs_cols'] = obs_cols
data_trajectory['acc_col']  = acc_cols
data_trajectory['rew_col'] = rew_col
data_trajectory['outcome_col'] = outcome_col
data_trajectory['obs_dim'] = len(obs_cols)
data_trajectory['traj'] = {}
data_trajectory['pos_traj'] = []
data_trajectory['neg_traj'] = []


for i in trajectories:
    # bar.update()
    traj_i = train_data[train_data['traj'] == i].sort_values(by='step')
    traj_j = train_acuity[train_acuity['traj']==i].sort_values(by='step')
    data_trajectory['traj'][i] = {}
    data_trajectory['traj'][i]['dem'] = torch.Tensor(traj_i[dem_cols].values).to('cpu')
    data_trajectory['traj'][i]['obs'] = torch.Tensor(traj_i[obs_cols].values).to('cpu')
    data_trajectory['traj'][i]['c_actions'] = torch.Tensor(traj_i[acc_cols].values.astype(np.float32)).to('cpu')
    data_trajectory['traj'][i]['rewards'] = torch.Tensor(traj_i[rew_col].values).to('cpu')
    data_trajectory['traj'][i]['acuity'] = torch.Tensor(traj_j[acuity_cols].values).to('cpu')
    data_trajectory['traj'][i]['outcomes'] = torch.Tensor(traj_i[outcome_col].values).to('cpu')
    
    
    if sum(traj_i[rew_col].values) > 0:
        data_trajectory['pos_traj'].append(i)
    else:
        data_trajectory['neg_traj'].append(i)


observations = torch.zeros((len(trajectories), horizon, num_obs))
demographics = torch.zeros((len(trajectories), horizon, num_dem)) 
c_actions = torch.zeros((len(trajectories), horizon-1, num_c_actions))
lengths = torch.zeros((len(trajectories)), dtype=torch.int)
times = torch.zeros((len(trajectories), horizon))
rewards = torch.zeros((len(trajectories), horizon))
acuities = torch.zeros((len(trajectories), horizon-1, num_acuity_scores))
outcomes = torch.zeros((len(trajectories), horizon))


for ii, traj in enumerate(trajectories):
    obs = data_trajectory['traj'][traj]['obs']
    dem = data_trajectory['traj'][traj]['dem']
    c_action = data_trajectory['traj'][traj]['c_actions'].view(-1,2)
    #print(torch.count_nonzero(c_action))
    reward = data_trajectory['traj'][traj]['rewards']
    #print(reward)
    outcome = data_trajectory['traj'][traj]['outcomes']
    acuity = data_trajectory['traj'][traj]['acuity']
    length = obs.shape[0]
    lengths[ii] = length
    observations[ii] = torch.cat((obs, torch.zeros((horizon-length, obs.shape[1]), dtype=torch.float)))
    demographics[ii] = torch.cat((dem, torch.zeros((horizon-length, dem.shape[1]), dtype=torch.float)))
    c_actions[ii] = torch.cat((c_action,torch.zeros((horizon-length-1, 2), dtype=torch.float)))
    times[ii] = torch.Tensor(range(horizon))
    rewards[ii] = torch.cat((reward, torch.zeros((horizon-length), dtype=torch.float)))
    outcomes[ii] = torch.cat((outcome, torch.zeros((horizon-length), dtype=torch.float)))
    acuities[ii] = torch.cat((acuity, torch.zeros((horizon-length-1, acuity.shape[1]), dtype=torch.float)))


# Eliminate single transition trajectories...
c_actions = c_actions[lengths>1.0].to(device)
observations = observations[lengths>1.0].to(device)
demographics = demographics[lengths>1.0].to(device)
times = times[lengths>1.0].to(device)
rewards = rewards[lengths>1.0].to(device)
outcomes = outcomes[lengths>1.0].to(device)
acuities = acuities[lengths>1.0].to(device)
lengths = lengths[lengths>1.0].to(device)

val_data = full_zs[full_zs['traj'].isin(X_val)]
val_acuity = acuity_scores[acuity_scores['traj'].isin(X_val)]
val_trajectories = val_data['traj'].unique()

## Validation DATA
#---------------------------------------------------------------------------
print("Converting Validation Data")
print("="*20)
val_data_trajectory = {}
val_data_trajectory['obs_cols'] = obs_cols
val_data_trajectory['dem_cols'] = dem_cols
val_data_trajectory['acc_col']  = acc_cols
val_data_trajectory['rew_col'] = rew_col
val_data_trajectory['outcome_col'] = outcome_col
val_data_trajectory['num_c_actions'] = num_c_actions
val_data_trajectory['obs_dim'] = len(obs_cols)
val_data_trajectory['traj'] = {}
val_data_trajectory['pos_traj'] = []
val_data_trajectory['neg_traj'] = []


for j in val_trajectories:
    traj_j = val_data[val_data['traj']==j].sort_values(by='step')
    traj_k = val_acuity[val_acuity['traj']==j].sort_values(by='step')
    val_data_trajectory['traj'][j] = {}
    val_data_trajectory['traj'][j]['dem'] = torch.Tensor(traj_j[dem_cols].values).to('cpu')
    val_data_trajectory['traj'][j]['obs'] = torch.Tensor(traj_j[obs_cols].values).to('cpu')
    val_data_trajectory['traj'][j]['c_actions'] = torch.Tensor(traj_j[acc_cols].values.astype(np.float32)).to('cpu')
    val_data_trajectory['traj'][j]['rewards'] = torch.Tensor(traj_j[rew_col].values).to('cpu')
    val_data_trajectory['traj'][j]['outcomes'] = torch.Tensor(traj_j[outcome_col].values).to('cpu')
    val_data_trajectory['traj'][j]['acuity'] = torch.Tensor(traj_k[acuity_cols].values).to('cpu')
    if sum(traj_j[rew_col].values) > 0:
        val_data_trajectory['pos_traj'].append(j)
    else:
        val_data_trajectory['neg_traj'].append(j)

val_obs = torch.zeros((len(val_trajectories), horizon, num_obs))
val_dem = torch.zeros((len(val_trajectories), horizon, num_dem))
val_c_actions = torch.zeros((len(val_trajectories), horizon-1, num_c_actions))
val_lengths = torch.zeros((len(val_trajectories)), dtype=torch.int)
val_times = torch.zeros((len(val_trajectories),horizon))
val_rewards = torch.zeros((len(val_trajectories), horizon))
val_outcomes = torch.zeros((len(val_trajectories), horizon))
val_acuities = torch.zeros((len(val_trajectories), horizon-1, num_acuity_scores))
action_temp = torch.eye(25)
for jj, traj in enumerate(val_trajectories):
    obs = val_data_trajectory['traj'][traj]['obs']
    dem = val_data_trajectory['traj'][traj]['dem']
    c_action = val_data_trajectory['traj'][traj]['c_actions'].view(-1,2)
    reward = val_data_trajectory['traj'][traj]['rewards']
    outcome = val_data_trajectory['traj'][traj]['outcomes']
    acuity = val_data_trajectory['traj'][traj]['acuity']
    length = obs.shape[0]
    val_lengths[jj] = length
    val_obs[jj] = torch.cat((obs, torch.zeros((horizon-length, obs.shape[1]), dtype=torch.float)))
    val_dem[jj] = torch.cat((dem, torch.zeros((horizon-length, dem.shape[1]), dtype=torch.float)))
    val_c_actions[jj] = torch.cat((c_action, torch.zeros((horizon-length-1, 2), dtype=torch.float)))
    val_times[jj] = torch.Tensor(range(horizon))
    val_rewards[jj] = torch.cat((reward, torch.zeros((horizon-length), dtype=torch.float)))
    val_outcomes[jj] = torch.cat((outcome, torch.zeros((horizon-length), dtype=torch.float)))
    val_acuities[jj] = torch.cat((acuity, torch.zeros((horizon-length-1, acuity.shape[1]), dtype=torch.float)))
# Eliminate single transition trajectories...
val_c_actions = val_c_actions[val_lengths>1.0].to(device)
val_obs = val_obs[val_lengths>1.0].to(device)
val_dem = val_dem[val_lengths>1.0].to(device)
val_times = val_times[val_lengths>1.0].to(device)
val_rewards = val_rewards[val_lengths>1.0].to(device)
val_outcomes = val_outcomes[val_lengths>1.0].to(device)
val_acuities = val_acuities[val_lengths>1.0].to(device)
val_lengths = val_lengths[val_lengths>1.0].to(device)


test_data = full_zs[full_zs['traj'].isin(X_test)]
test_acuity = acuity_scores[acuity_scores['traj'].isin(X_test)]
test_trajectories = test_data['traj'].unique()


## Test DATA
#---------------------------------------------------------------------------
print("Converting Test Data")
print("+"*20)
test_data_trajectory = {}
test_data_trajectory['obs_cols'] = obs_cols
test_data_trajectory['dem_cols'] = dem_cols
test_data_trajectory['acc_col']  = acc_cols
test_data_trajectory['rew_col'] = rew_col
test_data_trajectory['outcome_col'] = outcome_col
test_data_trajectory['num_c_actions'] = num_c_actions
test_data_trajectory['obs_dim'] = len(obs_cols)
test_data_trajectory['traj'] = {}
test_data_trajectory['pos_traj'] = []
test_data_trajectory['neg_traj'] = []

for j in test_trajectories:
    traj_j = test_data[test_data['traj']==j].sort_values(by='step')
    traj_k = test_acuity[test_acuity['traj']==j].sort_values(by='step')
    test_data_trajectory['traj'][j] = {}
    test_data_trajectory['traj'][j]['obs'] = torch.Tensor(traj_j[obs_cols].values).to('cpu')
    test_data_trajectory['traj'][j]['dem'] = torch.Tensor(traj_j[dem_cols].values).to('cpu')
    test_data_trajectory['traj'][j]['c_actions'] = torch.Tensor(traj_j[acc_cols].values.astype(np.float32)).to('cpu')
    test_data_trajectory['traj'][j]['rewards'] = torch.Tensor(traj_j[rew_col].values).to('cpu')
    test_data_trajectory['traj'][j]['outcomes'] = torch.Tensor(traj_j[outcome_col].values).to('cpu')
    test_data_trajectory['traj'][j]['acuity'] = torch.Tensor(traj_k[acuity_cols].values).to('cpu')
    if sum(traj_j[rew_col].values) > 0:
        test_data_trajectory['pos_traj'].append(j)
    else:
        test_data_trajectory['neg_traj'].append(j)

test_obs = torch.zeros((len(test_trajectories), horizon, num_obs))
test_dem = torch.zeros((len(test_trajectories), horizon, num_dem))
test_c_actions = torch.zeros((len(test_trajectories), horizon-1, num_c_actions))
test_lengths = torch.zeros((len(test_trajectories)), dtype=torch.int)
test_times = torch.zeros((len(test_trajectories), horizon))
test_rewards = torch.zeros((len(test_trajectories), horizon))
test_outcomes = torch.zeros((len(test_trajectories), horizon))
test_acuities = torch.zeros((len(test_trajectories), horizon-1, num_acuity_scores))
action_temp = torch.eye(25)
for jj, traj in enumerate(test_trajectories):
    obs = test_data_trajectory['traj'][traj]['obs']
    dem = test_data_trajectory['traj'][traj]['dem']
    c_action = test_data_trajectory['traj'][traj]['c_actions'].view(-1,2)
    reward = test_data_trajectory['traj'][traj]['rewards']
    outcome = test_data_trajectory['traj'][traj]['outcomes']
    acuity = test_data_trajectory['traj'][traj]['acuity']
    length = obs.shape[0]
    test_lengths[jj] = length
    test_obs[jj] = torch.cat((obs, torch.zeros((horizon-length, obs.shape[1]), dtype=torch.float)))
    test_dem[jj] = torch.cat((dem, torch.zeros((horizon-length, dem.shape[1]), dtype=torch.float)))
    test_c_actions[jj] = torch.cat((c_action, torch.zeros((horizon-length-1, 2), dtype=torch.float)))
    test_times[jj] = torch.Tensor(range(horizon))
    test_rewards[jj] = torch.cat((reward, torch.zeros((horizon-length), dtype=torch.float)))
    test_outcomes[jj] = torch.cat((outcome, torch.zeros((horizon-length), dtype=torch.float)))
    test_acuities[jj] = torch.cat((acuity, torch.zeros((horizon-length-1, acuity.shape[1]), dtype=torch.float)))

# Eliminate single transition trajectories...
test_c_actions = test_c_actions[test_lengths>1.0].to(device)
test_obs = test_obs[test_lengths>1.0].to(device)
test_dem = test_dem[test_lengths>1.0].to(device)
test_times = test_times[test_lengths>1.0].to(device)
test_acuities = test_acuities[test_lengths>1.0].to(device)
#test_mortality = test_mortality[test_lengths>1.0].to(device)
test_rewards = test_rewards[test_lengths>1.0].to(device)
test_outcomes = test_outcomes[test_lengths>1.0].to(device)
test_lengths = test_lengths[test_lengths>1.0].to(device)

#### Save off the tuples...
#############################
save_dir = '../data/'
train_file = 'train_set_tuples'
val_file = 'val_set_tuples'
test_file = 'test_set_tuples'
print("Saving off tuples")
print("..."*20)
torch.save((demographics,observations, c_actions, lengths,times,acuities,rewards, outcomes),os.path.join(save_dir,train_file))

torch.save((val_dem,val_obs,val_c_actions, val_lengths,val_times,val_acuities,val_rewards, val_outcomes),os.path.join(save_dir,val_file))

torch.save((test_dem,test_obs, test_c_actions, test_lengths,test_times,test_acuities,test_rewards, test_outcomes),os.path.join(save_dir,test_file))

print("\n")
print("Finished conversion")

# # We also extract and save off the mortality outcome of patients in the test set for evaluation and analysis purposes
# print("\n")
print("Extracting Test set mortality")
test_mortality = torch.Tensor(test_data.groupby('traj')['r:reward'].sum().values)
test_mortality = test_mortality.unsqueeze(1).unsqueeze(1)
test_mortality = test_mortality.repeat(1,20,1)  # Put in the same general format as the patient trajectories
# Save off mortality tuple
torch.save(test_mortality,os.path.join(save_dir,'test_mortality_tuple'))