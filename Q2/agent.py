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
from matplotlib import pyplot as plt


class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length
        self.episode_length = 400
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None

        self.step_size = 0.01
        self.disc_actions = [0, 1, 2, 3]
        self.cont_actions = self.step_size * np.array([
            [0.0, 1.0],  # North
            [1.0, 0.0],  # East
            [0.0, -1.0],  # South
            [-1.0, 0.0],  # West
        ], dtype=np.float32)
        self.disc_states_vector = np.arange(0, 1, self.step_size)

        # Eploration
        self.epsilon = 1
        self.decay_epsilon = True
        self.eps_dec_factor = 0.999
        self.min_eps = 0.1

        # Memory
        self.use_prioritisation = True
        self.Buffer = ReplayBuffer(self.use_prioritisation)
        self.use_online_learning = False
        self.mini_batch_size = 1 if self.use_online_learning else 100

        self.N_update_target = 15

        # Network
        self.dqn = DQN() 

        # Logging
        self.random_action_taken = []

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
        self.Buffer.append(transition)
        
        mini_batch_size = self.mini_batch_size if len(self.Buffer.buffer) > self.mini_batch_size else len(self.Buffer.buffer) # just for beginning
        mini_batch = self.Buffer.sample(sample_size=mini_batch_size)

        loss = self.dqn.train_q_network(mini_batch) 

        if self.decay_epsilon:
            self.decay_epsilon_strategy()

        if self.dqn.use_target and (self.num_steps_taken % self.N_update_target == 0):
            self.dqn.update_t_network()

        self.state = next_state

    def get_reward(self, distance_to_goal):
        return (1 - distance_to_goal)

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        # Here, the greedy action is fixed, but you should change it so that it returns the action with the highest Q-value
        action = np.array([0.02, 0.0], dtype=np.float32)
        return action

    def decay_epsilon_strategy(self):
        # dqn.epsilon = np.max((min_eps, dqn.epsilon-(1-eps_dec_factor)))
        self.epsilon = self.epsilon * self.eps_dec_factor

    def log(self):
        if not self.num_steps_taken:
            return

        if not self.random_action_taken:
            action_percs = 0
            total_rand = 0
        else:
            action_percs = []
            total_rand = np.sum(self.random_action_taken)
            for act in self.disc_actions:
                action_percs.append((act, (self.random_action_taken.count(act))/self.num_steps_taken))
        print('Random actions were taken {} perc. of the time: {}'.format(round(total_rand/self.num_steps_taken, 4), action_percs))

        self.dqn.losses.append(self.dqn.last_loss)
        self.log_loss()

    def log_loss(self):
        # Plot and save the loss vs iterations graph
        variance = np.std(self.dqn.losses)
        iterations = np.arange(0, len(self.dqn.losses))
        self.dqn.losses = [i if i is not None else 0 for i in self.dqn.losses]  # Remove 'None's

        text_eps = 'Epsilon decaying' if self.decay_epsilon else ''
        text_target = 'tnet' if self.dqn.use_target else 'qnet'
        text_loss_type = self.dqn.loss_type

        fig, ax = plt.subplots()
        ax.set(xlabel='Iteration', ylabel='Loss', title=('Loss Curve, Batch_size={} '+text_eps).format(self.mini_batch_size))
        ax.plot(iterations, self.dqn.losses, color='blue')
        plt.text(iterations[int(0.8*len(iterations))], self.dqn.losses[int(0.8*np.max(self.dqn.losses))], 'std={}'.format(variance))
        plt.yscale('log')
        fig.savefig("Q2/loss__batch{}_lr{}_epdec{}_epsteps{}_{}_{}.png".format(self.mini_batch_size, self.dqn.lr, self.eps_dec_factor, self.episode_length, text_loss_type, text_target))

    def log_q_values(self):
        
        # for 
        pass


class ReplayBuffer:

    def __init__(self, use_prioritisation=True, use_online_learning=False):
        self.buffer = deque(maxlen=5000)
        self.use_online_learning = use_online_learning
        self.use_prioritisation = use_prioritisation
        if use_prioritisation:
            self.buffer_probs = deque(maxlen=5000)
            self.default_prob = 0.1
            self.prob_factor = 1.5

    def update_weights(self, idx):
        for i in idx:
            self.buffer_probs[i] = self.min_p * self.prob_factor

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
                buffer_probs = np.array(self.buffer_probs) / np.sum(self.buffer_probs)
                random_indices = np.random.choice(list(range(buffer_size)), size=sample_size, replace=False, p=buffer_probs)
        else:
            random_indices = np.random.randint(0, buffer_size, size=sample_size)   # No probs

        # Collect
        all_s = []
        all_a = []
        all_r = []
        all_next_s = []
        for idx in random_indices:
            state, act, rew, next_state = self.buffer[idx]
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


class DQN:

    def __init__(self):
        # Architecture
        self.q_network = Network(input_dimension=2, output_dimension=4)
        self.t_network = Network(input_dimension=2, output_dimension=4)
        
        # Strategies
        self.use_target = True
        self.use_double_q = 'action'  # 'action': Use Q net for Q values, T net for action | 'values': Use T net for Q values, Q net for action
        self.loss_type = 'bellman'

        # HPs
        self.discount = 0.9
        self.lr = 0.002

        # Technical
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.loss_criterion = torch.nn.MSELoss()

        # Logging
        self.losses = []

    def update_t_network(self):
        q_weights = self.q_network.state_dict()
        self.t_network.load_state_dict(q_weights)

    def train_q_network(self, data):
        self.optimiser.zero_grad()

        loss = self._calculate_loss(data)
        loss.backward()

        self.optimiser.step()

        self.last_loss = loss.item()
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
        if self.use_double_q == 'action':
            q_prediction_ = self.q_network.forward(state_).detach()
            max_action = torch.unsqueeze(torch.argmax(prediction_, -1), 1)
            predicted_action_ = torch.gather(q_prediction_, -1, max_action)
        elif self.use_double_q == 'values':
            q_prediction_ = self.q_network.forward(state_).detach()
            max_action = torch.unsqueeze(torch.argmax(q_prediction_, -1), 1)
            predicted_action_ = torch.gather(prediction_, -1, max_action)
        else:
            # Q(S,A)
            predicted_action_ = prediction_.max(1)[0].detach().numpy()
        reward = torch.unsqueeze(reward, np.argmin(np.shape(predicted_action_)))

        # Bellman
        # --> R_Q = R + γ * max_a [ Q(S', a) ]
        prediction_reward = reward + self.discount * predicted_action_

        if self.loss_type == 'bellman':
            # --> θ = θ - α * (1/N) * sum[ R_Q - Q(S, A) ]^2
            loss = self.loss_criterion(torch.squeeze(prediction_reward), torch.squeeze(predicted_action))
        else:
            loss = self.loss_criterion(reward, predicted_action)
        return loss


  
def main():
    # TODO visualise Q values
    # TODO implement double dqn
    # TODO prioritisation
    # HP's
    hyperparameters = {
        'episode_length': 400
    }

    # AGENT 
    self.episode_length = 400
    self.step_size = 0.01

    # Eploration
    self.epsilon = 1
    self.decay_epsilon = True
    self.eps_dec_factor = 0.995
    self.min_eps = 0.1

    # Memory
    self.Buffer = ReplayBuffer()
    self.use_online_learning = False
    self.mini_batch_size = 1 if self.use_online_learning else 100

    self.N_update_target = 15

    # DQN
    # Strategies
    self.use_target = True
    self.loss_type = 'bellman'

    # HPs
    self.discount = 0.9
    self.lr = 0.002

    # Technical
    self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
    self.loss_criterion = torch.nn.MSELoss()

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