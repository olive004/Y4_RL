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

import numpy as np
import torch
from collections import deque
import copy
from matplotlib import pyplot as plt


class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length
        self.episode_length = 250
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None

        # Movement
        self.step_size = 0.01
        self.disc_actions = [0, 1, 2, 3]
        self.cont_actions = self.step_size * np.array([
            [0.0, 1.0],  # North
            [1.0, 0.0],  # East
            [0.0, -1.0],  # South
            [-1.0, 0.0],  # West
        ], dtype=np.float32)

        self.disc_states_vector = np.array(np.arange(0, 1, self.step_size), dtype=np.float32)
        self.q_values = np.zeros((len(self.disc_states_vector), len(self.disc_states_vector )))
        self.v_values = np.zeros((len(self.disc_states_vector), len(self.disc_states_vector )))

        # Eploration
        self.epsilon = 1
        self.decay_epsilon = True
        self.stop_at_min_eps = True
        self.eps_dec_factor = 0.999
        self.min_eps = 0.15

        # Strategy
        self.N_update_target = 15
        self.use_softupdate = True
        self.reward_type = 'linear'  # 'exponential'  # 

        self.use_online_learning = False
        self.mini_batch_size = 1 if self.use_online_learning else 64

        # Network
        self.dqn = DQN(self.mini_batch_size) 

        # Logging
        self.random_action_taken = []
        self.text_eps = 'Epsilon decaying' if self.decay_epsilon else ''
        self.text_target = self.dqn.use_double_q

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            self.log()
            return True
        else:
            return False

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        # Here, the action is random, but you can change this
        rand = np.random.rand()
        if rand < self.epsilon:
            # action = np.random.uniform(low=-self.step_size, high=self.step_size, size=2).astype(np.float32)
            action = np.random.choice(self.disc_actions)
            self.random_action_taken.append(action)
        else:
            predicted_rewards = self.dqn.q_network.forward(torch.tensor(self.state))
            action = np.argmax(predicted_rewards.detach().numpy())

        self.num_steps_taken += 1

        self.state = state
        self.action = action
        cont_action = self._discrete_action_to_continuous(action)

        return cont_action

    def _discrete_action_to_continuous(self, discrete_action):
        continuous_action = self.cont_actions[int(discrete_action)]
        return continuous_action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Convert the distance to a reward
        reward = self.get_reward(distance_to_goal)
        # Create a transition
        transition = (self.state, self.action, reward, next_state)
        # Now you can do something with this transition ...
        self.dqn.Buffer.append(transition)
        
        if self.dqn.Buffer.is_enough():
            mini_batch = self.dqn.Buffer.sample(sample_size=self.mini_batch_size)

            loss = self.dqn.train_q_network(mini_batch)

            if self.decay_epsilon:
                if self.stop_at_min_eps:
                    self.epsilon = np.max((self.min_eps, self.epsilon - (1-self.eps_dec_factor)))
                else:
                    self.decay_epsilon_strategy()

            if self.num_steps_taken % self.N_update_target == 0:
                if self.use_softupdate:
                    self.dqn.soft_update_t_network()
                else:
                    self.dqn.update_t_network()
                

            # if self.epsilon < (self.min_eps * 0.01):
            #     self.epsilon = 1

        self.state = next_state

    def get_reward(self, distance_to_goal):
        reward = (1 - distance_to_goal)
        if self.reward_type == 'linear':
            return reward
        if self.reward_type == 'exponential':
            if distance_to_goal < 0.2:
                reward *= 1.5
            elif distance_to_goal < 0.4:
                reward *= 1.3
            elif distance_to_goal < 0.7:
                reward *= 1.1
        return reward

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        # Here, the greedy action is fixed, but you should change it so that it returns the action with the highest Q-value
        # action = np.array([0.02, 0.0], dtype=np.float32)
        prediction = self.dqn.q_network.forward(torch.tensor(state)).detach()
        action = int(np.argmax(prediction.detach().numpy()))
        action = self._discrete_action_to_continuous(action)
        return action

    def decay_epsilon_strategy(self):
        # dqn.epsilon = np.max((min_eps, dqn.epsilon-(1-eps_dec_factor)))
        self.epsilon = self.epsilon * self.eps_dec_factor

    def log(self):
        if not(self.num_steps_taken and self.dqn.Buffer.is_enough()):
            return

        if not self.random_action_taken:
            action_percs = 0
            total_rand = 0
        else:
            action_percs = []
            total_rand = len(self.random_action_taken)
            for act in self.disc_actions:
                action_percs.append((act, (self.random_action_taken.count(act))/self.num_steps_taken))
        print('Random actions were taken {} perc. of the time: {}'.format(round(total_rand/self.num_steps_taken, 4), action_percs))
        self.dqn.losses.append(self.dqn.last_loss)
        self.log_loss()
        self.log_q_values()

    def log_loss(self):
        # Plot and save the loss vs iterations graph
        variance = np.std(self.dqn.losses)
        iterations = np.arange(0, len(self.dqn.losses))
        self.dqn.losses = [i if i is not None else 0 for i in self.dqn.losses]  # Remove 'None's

        fig, ax = plt.subplots()
        ax.set(xlabel='Iteration', ylabel='Loss', title=('Loss Curve, Batch_size={} '+self.text_eps).format(self.mini_batch_size))
        ax.plot(iterations, self.dqn.losses, color='blue')
        plt.text((0.8*len(iterations)), (0.8*np.max(self.dqn.losses)), 'std={}'.format(variance))
        plt.yscale('log')
        fig.savefig("Q2/loss__bs{}_lr{}_disc{}_epdec{}_epsteps{}_tau{}_N{}_{}.png".format(self.mini_batch_size, self.dqn.lr, self.dqn.discount, self.eps_dec_factor, self.episode_length, self.dqn.tau, self.N_update_target, self.text_target))

    def log_q_values(self):
        
        for i, sx in enumerate(self.disc_states_vector):
            for j, sy in enumerate(self.disc_states_vector):
                prediction = self.dqn.q_network.forward(torch.tensor([sx, sy])).detach().numpy()
                self.q_values[i, j] = np.max(prediction)
                # self.v_values[i, j] = np.mean(prediction)
        fig, ax = plt.subplots()
        im = ax.imshow(np.flip(self.q_values, axis=0))
        fig.colorbar(im)
        fig.savefig("Q2/QValues__bs{}_lr{}_disc{}_epdec{}_epsteps{}_tau{}_N{}_{}.png".format(self.mini_batch_size, self.dqn.lr, self.dqn.discount, self.eps_dec_factor, self.episode_length, self.dqn.tau, self.N_update_target, self.text_target))

        # fig, ax = plt.subplots()
        # im = ax.imshow(np.flip(self.v_values, axis=0))
        # fig.colorbar(im)
        # fig.savefig("Q2/VValues__bs{}_lr{}_disc{}_epdec{}_epsteps{}_tau{}_N{}_{}.png".format(self.mini_batch_size, self.dqn.lr, self.dqn.discount, self.eps_dec_factor, self.episode_length, self.dqn.tau, self.N_update_target, self.text_target))


class ReplayBuffer:

    def __init__(self, sample_size=100, maxlen=5000, use_prioritisation=True, use_penalisation=True, use_importance_sampling=True, use_online_learning=False):
        self.buffer = deque(maxlen=maxlen)
        self.sample_size = sample_size
        self.use_online_learning = use_online_learning
        self.use_prioritisation = use_prioritisation
        self.use_importance_sampling = use_importance_sampling
        if use_prioritisation:
            self.buffer_probs = deque(maxlen=maxlen)
            self.default_prob = 0.01
            self.alpha = 1
            self.alpha_decay = 0.99
            self.recent_idxs = []
        if use_importance_sampling:
            self.beta = 0.4
            self.weights = []
        self.use_penalisation = use_penalisation

    def is_enough(self):
        if len(self.buffer) > self.sample_size:
            return True
        else:
            return False

    def update_priority(self, weights):
        weights = weights if (np.size(weights)>1) else [weights]
        for i, w in zip(self.recent_idxs, weights):
            self.buffer_probs[i] = self.default_prob + np.abs(w)

    def sample(self, sample_size=20):

        if self.use_online_learning:
            state, act, rew, next_state = self.buffer[-1]
            return state, act, rew, next_state

        buffer_size = len(self.buffer)
        sample_size = sample_size if buffer_size > sample_size else buffer_size
        
        # Choose 'sample_size' number of random indices based on probabilities in buffer_probs
        if self.use_prioritisation:
            total_prob = np.sum(self.buffer_probs)
            if total_prob:
                buffer_probs = (np.array(self.buffer_probs) ** self.alpha)
                buffer_probs =  buffer_probs / np.sum(buffer_probs)
                random_indices = np.random.choice(list(range(buffer_size)), size=sample_size, replace=False, p=buffer_probs)

                self.alpha *= self.alpha_decay
        else:
            random_indices = np.random.randint(0, buffer_size, size=sample_size)   # No probs
        np.sort(random_indices)
        self.recent_idxs = random_indices

        # Collect
        all_s = []
        all_a = []
        all_r = []
        all_next_s = []
        for idx in random_indices:
            state, act, rew, next_state = self.buffer[idx]
            if self.use_penalisation and np.all(state == next_state):
                rew = rew / 1.5
            all_s.append(state)
            all_a.append(act)
            all_r.append(rew)
            all_next_s.append(next_state)

        return all_s, all_a, all_r, all_next_s

    def append(self, transition):
        self.buffer.append(transition)
        if self.use_prioritisation:
            self.buffer_probs.append(self.default_prob)


class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


class DuelingNetwork(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension):
        super(DuelingNetwork, self).__init__()


class DQN:

    def __init__(self, mini_batch_size):
        # Architecture
        self.q_network = Network(input_dimension=2, output_dimension=4)
        self.t_network = copy.deepcopy(self.q_network)
        # self.t_network = Network(input_dimension=2, output_dimension=4)
        
        # Strategies
        self.use_nstep_discount = False
        self.use_double_q = 'values'  # 'target': classic target network | 'action': Use Q net for Q values, T net for action | 'values': Use T net for Q values, Q net for action
        self.loss_type = 'bellman'

        # HPs
        self.discount = 0.95
        self.discount_vector = torch.tensor([self.discount ** i for i in range(mini_batch_size)])
        self.lr = 0.001
        self.tau = 0.4

        # Memory
        self.use_penalisation = True
        self.use_prioritisation = False
        self.use_importance_sampling = False
        maxlen = 20000
        self.Buffer = ReplayBuffer(mini_batch_size, maxlen, self.use_prioritisation, self.use_penalisation, self.use_importance_sampling)

        # Technical
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.loss_criterion = torch.nn.MSELoss()

        # Logging
        self.losses = []
        self.last_loss = None

    def update_t_network(self):
        q_weights = self.q_network.state_dict()
        self.t_network.load_state_dict(q_weights)

    def soft_update_t_network(self):
        for t_param, q_param in zip(self.t_network.parameters(), self.q_network.parameters()):
            t_param.data.copy_(self.tau * q_param.data + (1.0-self.tau)*t_param.data)

    def train_q_network(self, data):
        self.optimiser.zero_grad()

        loss = self._calculate_loss(data)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1)
        self.optimiser.step()

        self.last_loss = loss.item()  # for logging
        return loss.item()

    def _calculate_loss(self, data_batch):
        # Input: S, A
        state, action, reward, state_ = data_batch
        reward = torch.tensor(reward).type(torch.float32)
        action = torch.unsqueeze(torch.tensor(action), 1)

        # Convert to Torch tensor
        state = torch.tensor(state)
        state_ = torch.tensor(state_)

        # Forward
        prediction = self.q_network.forward(state)
        prediction_ = self.t_network.forward(state_).detach()
        
        predicted_action = torch.gather(prediction, -1, action)
        q_prediction_ = self.q_network.forward(state_).detach()
        if self.use_double_q == 'action':
            max_action = torch.unsqueeze(torch.argmax(prediction_, -1), 1)
            predicted_action_ = torch.gather(q_prediction_, -1, max_action)
        elif self.use_double_q == 'values':
            max_action = torch.unsqueeze(torch.argmax(q_prediction_, -1), 1)
            predicted_action_ = torch.gather(prediction_, -1, max_action)
        else:
            # Q(S,A)
            predicted_action_ = prediction_.max(1)[0].detach()  #.numpy()

        if self.use_nstep_discount:
            discount = self.discount_vector[0:len(predicted_action_)]
        else:
            discount = self.discount

        # Bellman
        # --> G = R + γ * max_a [ Q(S', a) ]
        prediction_reward = reward + discount * torch.squeeze(predicted_action_)

        # Buffer update
        if self.use_prioritisation:
            buffer_idxs = self.Buffer.recent_idxs[(prediction_reward > torch.mean(
                prediction_reward, axis=0) + torch.std(prediction_reward, axis=0)).squeeze()]
            buffer_weights = ((torch.squeeze(prediction_reward) - torch.squeeze(predicted_action)) ** 2).detach().numpy()
            self.Buffer.update_priority(weights=buffer_weights)

        if self.loss_type == 'bellman':
            # --> θ = θ - α * (1/N) * sum[ G - Q(S, A) ]^2
            loss = self.loss_criterion(torch.squeeze(prediction_reward), torch.squeeze(predicted_action))
        else:
            loss = self.loss_criterion(reward, predicted_action)
        return loss



def main():
    # TODO penalise wall bumping -> q values exploding
    # TODO N-step discount 
    # TODO fix prioritisation
    # TODO intrinsic motivation
    # TODO dueling heads
    # TODO HP's
    # TODO importance sampling
    hyperparameters = {
        'episode_length': 250,
        'eps decay': 0.999,
        'replay buffer max': 5000,
        'minibatch': 100
    }


    # AGENT
    # Movement
    self.step_size = 0.01
    self.disc_actions = [0, 1, 2, 3]
    self.cont_actions = self.step_size * np.array([
        [0.0, 1.0],  # North
        [1.0, 0.0],  # East
        [0.0, -1.0],  # South
        [-1.0, 0.0],  # West
    ], dtype=np.float32)

    self.disc_states_vector = np.array(np.arange(0, 1, self.step_size), dtype=np.float32)
    self.q_values = np.zeros((len(self.disc_states_vector), len(self.disc_states_vector )))

    # Eploration
    self.epsilon = 1
    self.decay_epsilon = True
    self.eps_dec_factor = 0.995

    # Strategy
    self.N_update_target = 30
    self.reward_type = 'linear'  # 'exponential'  # 
    self.use_penalisation = False

    # Memory
    self.use_prioritisation = True
    self.Buffer = ReplayBuffer(self.use_prioritisation, self.use_penalisation)
    self.mini_batch_size = 1 if self.use_online_learning else 100
    
    # NETWORK
    # Strategies
    self.use_nstep_discount = False
    self.use_double_q = 'action'  # 'target': classic target network | 'action': Use Q net for Q values, T net for action | 'values': Use T net for Q values, Q net for action

    # HPs
    self.discount = 0.9
    self.lr = 0.0002



    # ep 98, bs 42, epdec 0.9, gamma 0.12884, action 4, ddqn, alpha 0.11, prioritised her, softmax exploration, 
# masturbation 
# what's a good lie to tell
#     Pretending to be in a course you're not
# who is the best boy 
# What's an ok time to come late
#     1 // 2 // 3 // 4 hrs
# Rate these tik toks
#     tik tok 1 // tik tok 2
# Who brings the party

# TD Error
# gamma * next_state
# TD Error next level: N step 
# gamma^n * np.sum(transitions)