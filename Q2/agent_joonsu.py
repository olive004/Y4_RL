############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################

import collections

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

class Agent(object):

    # Function to initialise the agent
    def __init__(self):

        # Set the episode length 
        self.episode_length = 350
        
        # Set training config
        double = True
        dueling = True
        noisy = False
        curiosity = False

        # Set the network config
        self.dqn = DQN(double=double,
                       dueling=dueling,
                       noisy=noisy,
                       curiosity=curiosity)
        
        # Exploration strategy
        if noisy:
            self.epsilon = 0
            self.min_epsilon = 0
            self.delta = 0
        else:
            self.epsilon = 1.0
            self.min_epsilon = 0.15
            self.delta = 1.5e-05

        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # Episodes count
        self.episodes = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # Store the initial state of the agent
        self.init_state = None
        # The action variable stores the latest action which the agent has taken in the environment
        self.action = None

        # Define the action space and step size of the agent
        self.action_space = np.array([0, 1, 2, 3])
        self.step_size = 0.02
        self.continuous_actions = self.step_size * np.array([
            [0, 1],
            [1, 0],
            [0, -1],
            [-1, 0]
        ], dtype=np.float32)

        # Periodically evaluate the optimal policy of the agent
        self.eval_mode = False

        # debug flags
        self.debug = False
        self.debug_optimal_policy = False

    # Check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            # Increment episode count
            self.episodes += 1
            # Check that evaluation is finished
            self.eval_mode = False
            # Train the model
            self.dqn.q_network.train()
            return True
        else:
            return False

    # Get next action given current state
    def get_next_action(self, state):
        # Print debugging info
        if self.debug:
            if self.num_steps_taken % 1000 == 0:
                print('Steps: {}, epsilon: {}, episode length: {}'.format(
                    self.num_steps_taken,
                    self.epsilon,
                    self.episode_length
                ))

        # Periodically evaluate the policy
        if self.policy_evaluation_conditions():
            self.eval_mode = True
            self.dqn.q_network.eval()

        # Increment the number of steps taken by the agent
        self.num_steps_taken += 1

        # Evaluate the policy
        if self.eval_mode:
            return self.get_greedy_action_eval(state)

        # Decrease the episode length so that it converges to 100 (number of steps during test time)
        if self.num_steps_taken > 10000 and  self.episodes % 10 == 0 and self.num_steps_taken % self.episode_length == 1:
            self.decrease_episode_length(delta=25)

        self.state = state

        # Get action
        action = self.epsilon_greedy_action()
        # Get the continuous action
        continuous_action = self.discrete2continuous(action)
        self.action = action
        return continuous_action

    # Set the next state and distance from action=self.action at state=self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Don't train the network during evaluation
        if self.eval_mode:
            if self.debug_optimal_policy:
                print('Evaluating greedy policy...')
                if distance_to_goal < 0.03:            
                    print('Greedy policy works! Distance from goal: {}.'.format(
                        distance_to_goal
                    ))                
                else:
                    print('Did not reach goal. Final distance = ' + str(distance_to_goal))
            return

        # Get the reward from the distance to goal
        reward = self.calculate_reward(next_state, distance_to_goal)
        # Create a transition
        transition = (self.state, self.action, reward, next_state)

        # Add transition to the buffer
        self.dqn.replay_buffer.add(transition)
        if self.dqn.replay_buffer.is_big_enough():
            self.dqn.train_q_network()
            self.epsilon = max(self.epsilon-self.delta, self.min_epsilon)

        # Update target network every 50th step
        if self.num_steps_taken % 50 == 0:
            self.dqn.update_target_network()

    # Function to calculate the reward
    def calculate_reward(self, next_state, distance_to_goal):
        reward = float(0.1*(1 - distance_to_goal))
        
        # Scale the reward exponentially w.r.t distance to goal
        if distance_to_goal <= 0.2:
            reward *= 3
        elif distance_to_goal <= 0.3:
            reward *= 2
        elif distance_to_goal <= 0.5:
            reward *= 1.5

        # If the agent does not move, penalise reward
        if not np.any(self.state - next_state):
            reward /= 1.5
        return reward

    # Get the greedy action for a particular state
    def get_greedy_action(self, state):
        action_rewards = self.dqn.q_network.forward(
            torch.tensor(state)
        ).detach().numpy()
        discrete_action = np.argmax(action_rewards)
        return self.discrete2continuous(discrete_action)

    # Get the greedy action for a particular state during evaluation
    def get_greedy_action_eval(self, state):
        action_rewards = self.dqn.q_network.forward(
            torch.tensor(state)
        ).detach().numpy()
        discrete_action = np.argmax(action_rewards)
        return self.discrete2continuous(discrete_action)
    
    # Choose random action
    def random_action(self):
        return self.action_space[np.random.randint(low=0, high=3)]

    # Epsilon-greedy action selection
    def epsilon_greedy_action(self):
        action_rewards = self.dqn.q_network.forward(
            torch.tensor(self.state)
        ).detach().numpy()
        prob = np.random.uniform(low=0.0, high=1.0)
        if prob < self.epsilon:
            return self.random_action()
        else:
            return np.argmax(action_rewards)
    
    # Convert discrete actions to continuous values
    def discrete2continuous(self, discrete_action):
        return self.continuous_actions[discrete_action]
    
    # Decrease episode length until it reaches minimum value of 100
    def decrease_episode_length(self, delta=50):
        if self.episode_length > 100:
            self.episode_length -= delta

    # check if policy should be evaluated
    def policy_evaluation_conditions(self):
        return all([
            self.episode_length == 100,
            self.num_steps_taken > 15000,
            self.episodes % 10 == 0,
            self.num_steps_taken % self.episode_length == 0
        ])

# NoisyNet Linear Layer
class NoisyLinear(nn.Module):
    # Implementation based on https://github.com/Scitator/Run-Skeleton-Run/blob/master/common/modules/NoisyLinear.py
    def __init__(self,
                 in_features,
                 out_features,
                 use_bias=True,
                 use_factorised=True,
                 std_init=None):
        super(NoisyLinear, self).__init__()
        self.input_dim = in_features
        self.output_dim = out_features
        self.use_factorised = use_factorised
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.use_bias = use_bias

        if use_bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        if not std_init:
            if self.use_factorised:
                self.std_init = 0.5
            else:
                self.std_init = 0.017
        else:
            self.std_init = std_init
        self.reset_parameters()

    def reset_parameters(self):
        # use factorised Gaussian distribution
        if self.use_factorised:
            mu_range = 1. / math.sqrt(self.weight_mu.size(1))
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
            if self.use_bias:
                self.bias_mu.data.uniform_(-mu_range, mu_range)
                self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
        # else use independent Gaussian distribution (more computationally expensive)
        else:
            mu_range = math.sqrt(3. / self.weight_mu.size(1))
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init)
            if self.use_bias:
                self.bias_mu.data.uniform_(-mu_range, mu_range)
                self.bias_sigma.data.fill_(self.std_init)

    def scale_noise(self, size):
        x = torch.Tensor(size).normal_()
        x = x.sign().mul(x.abs().sqrt())
        return x

    def forward(self, input):
        if self.use_factorised:
            epsilon_in = self.scale_noise(self.input_dim)
            epsilon_out = self.scale_noise(self.output_dim)
            weight_epsilon = Variable(epsilon_out.ger(epsilon_in))
            bias_epsilon = Variable(self.scale_noise(self.output_dim))
        else:
            weight_epsilon = Variable(torch.Tensor(self.output_dim, self.input_dim).normal_())
            bias_epsilon = Variable(torch.Tensor(self.output_dim).normal_())
        return F.linear(input,
                        self.weight_mu + self.weight_sigma.mul(weight_epsilon),
                        self.bias_mu + self.bias_sigma.mul(bias_epsilon))

# Default MLP Architecture for vanilla DQN
class Network(nn.Module):

    def __init__(self,
                 input_dimension,
                 output_dimension,
                 noisy=False):
        super(Network, self).__init__()

        self.noisy = noisy

        if noisy:
            linear = NoisyLinear
        else:
            linear = nn.Linear
        
        self.layer_1 = nn.Linear(
            in_features=input_dimension, out_features=100)
        self.layer_2 = linear(
            in_features=100, out_features=100)
        self.output_layer = linear(
            in_features=100, out_features=output_dimension)

    def forward(self, input):
        layer_1_output = F.relu(self.layer_1(input))
        layer_2_output = F.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output
    
    def reset_parameters(self):
        assert self.noisy == True
        self.layer_2.reset_parameters()
        self.output_layer.reset_parameters()

# Dueling Network Architecture
class DuelingNetwork(nn.Module):
    
    def __init__(self,
                 input_dimension,
                 output_dimension,
                 noisy=False):
        super(DuelingNetwork, self).__init__()

        self.noisy = noisy

        if noisy:
            linear = NoisyLinear
        else:
            linear = nn.Linear

        # Common input layer
        self.layer_1 = nn.Linear(
            in_features=input_dimension, out_features=100)
        
        # Value function head
        self.layer_value = linear(
            in_features=100, out_features=100)
        self.output_value = linear(
            in_features=100, out_features=1)
        
        # Advantage function head
        self.layer_adv = linear(
            in_features=100, out_features=100)
        self.output_adv = linear(
            in_features=100, out_features=output_dimension) 

    def forward(self, state):
        y = F.relu(self.layer_1(state))
        value = F.relu(self.layer_value(y))
        adv = F.relu(self.layer_adv(y))

        value = self.output_value(value)
        adv = self.output_adv(adv)

        output = value + adv - adv.mean()
        return output

    def reset_parameters(self):
        assert self.noisy == True
        self.layer_value.reset_parameters()
        self.output_adv.reset_parameters()
        self.layer_adv.reset_parameters()
        self.output_value.reset_parameters()

# Main DQN agent
class DQN(object):

    def __init__(self,
                 double=False,
                 dueling=False,
                 noisy=False,
                 curiosity=False
                 ):
        
        if dueling:
            self.q_network = DuelingNetwork(input_dimension=2,
                                            output_dimension=4,
                                            noisy=noisy)
            self.target_q_network = DuelingNetwork(input_dimension=2,
                                                   output_dimension=4,
                                                   noisy=noisy)
        else:
            self.q_network = Network(input_dimension=2,
                                     output_dimension=4,
                                     noisy=noisy)
            self.target_q_network = Network(input_dimension=2,
                                            output_dimension=4,
                                            noisy=noisy)

        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=2.5e-03)
        self.replay_buffer = ReplayBuffer()
        self.discount_factor = 0.95
        self.double = double
        self.curiosity = curiosity

        if self.curiosity:
            inverse_model = Inverse(2, 4)
            forward_model = Forward(2, 4, inverse_model.compute_input_layer())
            self.ICM = ICM(
                inverse_model,
                forward_model
            )

    def train_q_network(self):
        self.optimiser.zero_grad()
        batch = self.replay_buffer.random_sample()
        loss = self._calculate_long_run_loss(batch)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(),1)
        self.optimiser.step()
        return loss.item()

    def _calculate_long_run_loss(self, batch):
        s, a, r, s_p, idx = batch

        if self.curiosity:
            forward_pred_err, inverse_pred_err = self.ICM.compute_errors(s=s, s_p=s_p, a=a)
            self.ICM.update_ICM(forward_pred_err, inverse_pred_err)
            # append intrinsic reward from ICM 
            r += torch.tensor(forward_pred_err, dtype=torch.float32).detach().numpy()
        
        predicted_rewards = self.q_network.forward(torch.tensor(s, dtype=torch.float32))
        prediction_tensor = torch.gather(predicted_rewards, 1, torch.tensor(a, dtype=torch.int64))

        if self.double:
            # compute the Q values for best action from current q network
            # max(Q(s', a', theta_i)) wrt a'
            argmax_predicted_rewards_prime = self.q_network.forward(
            torch.tensor(s_p, dtype=torch.float32)).argmax(1).detach().reshape(-1, 1)
            
            # compute Q values from target network for next state and chosen action
            # Q(s',argmax(Q(s',a', theta_i), theta_i_target)) (argmax wrt a')
            predicted_rewards_prime = self.target_q_network.forward(
            torch.tensor(s_p, dtype=torch.float32)).gather(1, argmax_predicted_rewards_prime).detach()
            
            # compute the Bellman error
            expected_value = r + self.discount_factor * predicted_rewards_prime.data.numpy()
        else:
            # compute the Q values for best action from target q network
            # max(Q(s', a', theta_i_target)) wrt a'
            predicted_rewards_prime = self.target_q_network.forward(
                torch.tensor(s_p, dtype=torch.float32)).detach()
            max_actions = np.argmax(
                predicted_rewards_prime.detach().numpy(), axis=1).reshape(-1, 1)
            state_prime_tensor = torch.gather(
                predicted_rewards_prime, 1, torch.tensor(max_actions)).detach()
            
            # compute the bellman error
            expected_value = r + self.discount_factor * state_prime_tensor.data.numpy()
        
        # compute the indices to update in the replay buffer
        idx_to_update = idx[(expected_value > np.mean(
            expected_value, axis=0) + np.std(expected_value, axis=0)).squeeze()]

        # update the weights in the replay buffer
        self.replay_buffer.update_weights(idx_to_update)
        
        return torch.nn.MSELoss()(torch.tensor(expected_value, dtype=torch.float32), prediction_tensor)

    def update_target_network(self):
        weights = self.q_network.state_dict()
        self.target_q_network.load_state_dict(weights)

# A Simplified Prioritised Replay Buffer
class ReplayBuffer(object):

    def __init__(self):
        # buffer and prioritisation weights
        self.buffer = collections.deque(maxlen=10000)
        self.p = collections.deque(maxlen=10000)
        # replay buffer sample size
        self.sample_size = 200
        # prioritisation value for idx where q value is overestimated
        self.min_p = 0.1

    def size(self):
        return len(self.buffer)

    def is_big_enough(self):
        return self.size() >= self.sample_size

    def add(self, transition):
        self.buffer.appendleft(transition)
        self.p.appendleft(self.min_p)

    def update_weights(self, idx):
        for i in idx:
            self.p[i] = self.min_p

    def random_sample(self):
        buffer_size = self.size()
        prob = np.array(self.p)
        prob = prob / np.sum(prob)
        sample_idx = np.random.choice(
            np.arange(buffer_size), size=self.sample_size, replace=False, p=prob)

        states = []
        actions = []
        rewards = []
        states_prime = []

        for idx in sample_idx:
            s, a, r, s_p = self.buffer[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            states_prime.append(s_p)

        states = np.array(states, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float64).reshape(-1, 1)
        actions = np.array(actions, dtype=np.int64).reshape(-1, 1)
        states_prime = np.array(states_prime, dtype=np.float32)

        return states, actions, rewards, states_prime, sample_idx

# Inverse model that predicts which action was taken given two states s and s'
class Inverse(nn.Module):
    def __init__(self,
                 state_size,
                 action_size,
                 hidden_size=100):
        super(Inverse, self).__init__()
        self.state_size = state_size
        
        self.encoder = nn.Sequential(nn.Linear(state_size, hidden_size),
                                     nn.ReLU())
        self.layer1 = nn.Linear(2*hidden_size, hidden_size)

        self.layer2 = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=1)
    
    def compute_input_layer(self):
        x = torch.zeros(self.state_size).unsqueeze(0)
        x = self.encoder(x)
        return x.flatten().shape[0]
    
    def forward(self, enc_state, enc_state_p):
        x = torch.cat((enc_state, enc_state_p), dim=1)
        x = torch.relu(self.layer1(x))
        x = self.softmax(self.layer2(x))
        return x

# Forward prediction network 
# which predicts the next encoded state phi' given the encoded current state phi  
class Forward(nn.Module):
    def __init__(self,
                 state_size,
                 action_size,
                 output_size,
                 hidden_size=100):
        super(Forward, self).__init__()
        self.action_size = action_size
        self.forward_model = nn.Sequential(nn.Linear(output_size+self.action_size,hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size,output_size))
    
    def forward(self, state, action):
        # One-hot-encoded actions 
        one_hot_enc_action = torch.zeros(action.shape[0], self.action_size)
        indices = torch.stack((torch.arange(action.shape[0]), action.squeeze().long()), dim=0)
        indices = indices.tolist()
        one_hot_enc_action[indices] = 1.
        # concatenate state and one-hot-encoded action
        x = torch.cat((state, one_hot_enc_action) ,dim=1)
        return self.forward_model(x)

# Intrinsic Curiosity Module
class ICM(nn.Module):
    def __init__(self,
                 inverse_model,
                 forward_model,
                 lr=1e-3,
                 beta=0.2):
        super(ICM, self).__init__()
        self.inverse_model = inverse_model
        self.forward_model = forward_model

        self.beta = beta
        self.forward_loss = torch.nn.MSELoss(reduction='none')
        self.inverse_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.param_list = list(self.forward_model.parameters()) + list(self.inverse_model.parameters())
        self.optimizer = torch.optim.Adam(self.param_list,lr=lr)

    def compute_errors(self,
                       s,
                       s_p,
                       a):

        enc_s = self.inverse_model.encoder(
            torch.tensor(s)).view(s.shape[0],-1)
        enc_s_p = self.inverse_model.encoder(
            torch.tensor(s_p)).view(s.shape[0],-1)

        # compute forward error 
        forward_pred = self.forward_model(
            enc_s.detach(), torch.tensor(a))
        forward_pred_err = self.forward_loss(
            forward_pred, enc_s_p.detach()).sum(dim=1).unsqueeze(dim=1)
        
        # compute prediction error
        pred_a = self.inverse_model(
            enc_s, enc_s_p) 
        inverse_pred_err = self.inverse_loss(
            pred_a, torch.tensor(a).flatten().long()).unsqueeze(dim=1)    
      
        return forward_pred_err, inverse_pred_err

    def update_ICM(self, forward_err, inverse_err):
        self.optimizer.zero_grad()
        loss = ((1. - self.beta)*inverse_err + self.beta*forward_err).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(self.inverse_model.parameters(),1)
        nn.utils.clip_grad_norm_(self.forward_model.parameters(),1)
        self.optimizer.step()
        return loss

