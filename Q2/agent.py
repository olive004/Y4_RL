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
    def __init__(self, 
        buffer_alpha_dec = 0,  # set prob to 0.1 if 
        eta = 0.1,
        eval_mode=False,
        discount=0.94,
        episode_length=350,  # reduce during training
        eps_dec_factor=(1-(2e-05)),
        lr=2e-03,
        min_eps=0.15,
        mini_batch_size=220, 
        N_update_target=50,
        network_type = 'dueling',
        num_random_epochs = 2,
        reward_type='exponential',
        reduce_episodes=True,
        stop_at_min_eps=True,
        tau=1,  # 1e-3,
        torch_seed=1,
        use_alt_epsilon=False,
        use_curiosity=0,    # can be 0, 1 or else
        use_diagonal_actions = False,
        use_double_q='values',
        use_experimental_action=False,
        use_penalisation=True,
        use_prioritisation=False,
        use_softupdate=False
    ):
        self.eval_mode = eval_mode
        # Set the episode length
        self.episode_length = episode_length
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None

        # Movement
        self.step_size = 0.02
        self.disc_actions = [0, 1, 2, 3]
        self.cont_actions = self.step_size * np.array([
            [0.0, 1.0],  # North
            [1.0, 0.0],  # East
            [0.0, -1.0],  # South
            [-1.0, 0.0],  # West
        ], dtype=np.float32)
        if use_diagonal_actions:
            self.step_size = 0.012
            self.cont_actions = self.step_size * np.array([
                [1.0, 1.0],  # North East
                [-1.0, 1.0],  # South East
                [1.0, -1.0],  # North West
                [-1.0, -1.0],  # South West
            ], dtype=np.float32)
        self.use_experimental_action = use_experimental_action

        self.disc_states_vector = np.array(np.arange(0, 1, self.step_size), dtype=np.float32)
        self.q_values = np.zeros((len(self.disc_states_vector), len(self.disc_states_vector)))
        self.v1_values = np.zeros((len(self.disc_states_vector), len(self.disc_states_vector)))
        self.v2_values = np.zeros((len(self.disc_states_vector), len(self.disc_states_vector)))
        self.v3_values = np.zeros((len(self.disc_states_vector), len(self.disc_states_vector)))

        # Eploration
        self.epsilon = 1.0
        self.alt_epsilon = self.epsilon
        self.decay_epsilon = True
        self.stop_at_min_eps = stop_at_min_eps
        self.eps_dec_factor = eps_dec_factor
        self.min_eps = min_eps
        self.use_curiosity = use_curiosity
        self.eta = eta
        self.num_random_epochs = num_random_epochs
        self.use_alt_epsilon = use_alt_epsilon

        # Strategy
        self.N_update_target = N_update_target
        self.use_softupdate = use_softupdate
        self.reward_type = reward_type  # 'linear'  # 'exponential'  # 
        self.reduce_episodes = reduce_episodes

        self.use_online_learning = False
        self.mini_batch_size = mini_batch_size

        # Network
        self.dqn = DQN(buffer_alpha_dec, discount, eta, lr, mini_batch_size, network_type, tau, torch_seed, use_curiosity, use_double_q, use_penalisation, use_prioritisation) 

        # Logging
        self.episode_count = 0
        self.visualise = False
        self.random_action_taken = []
        self.text_eps = 'Epsilon decaying' if self.decay_epsilon else ''
        self.text_target = self.dqn.use_double_q
        self.per_text = '_per{}'.format(str(self.dqn.Buffer.alpha_decay)) if use_prioritisation else ''
        self.text_net = 'net{}'.format(network_type)
        self.test_icm = '_icm' if use_curiosity else ''
        self.text_episodes = '_reducEpisodes' if self.reduce_episodes else ''
        self.text_alt_eps = '_alteps' if use_alt_epsilon else ''
        self.text_penl = '_noPen' if use_penalisation else ''
        self.fig_text = "{}_bs{}_lr{}_disc{}_epdec{}{}_epsteps{}{}{}_tau{}_N{}_{}{}{}".format(self.text_net, self.mini_batch_size, self.dqn.lr, self.dqn.discount, self.eps_dec_factor, self.text_penl, self.episode_length, self.text_episodes, self.text_alt_eps, self.dqn.tau, self.N_update_target, self.text_target, self.per_text, self.test_icm)

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            if self.eval_mode and self.visualise:
                self.log()
            # self.epsilon = np.min((1, self.epsilon + (1-self.eps_dec_factor)*(self.episode_length/4)))

            self.episode_count +=1 
            return True
        else:
            return False

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        if self.reduce_episodes and (self.episode_count % 10 == 0) and (self.episode_count > 80):
            self.episode_length = int(np.max((100, self.episode_length-25)))

        # Here, the action is random, but you can change this
        rand = np.random.uniform(low=0, high=1)
        # epsilon = self.alt_epsilon if self.dqn.Buffer.use_alt_epsilon else self.epsilon
        if self.use_alt_epsilon:
            if self.dqn.Buffer.use_alt_epsilon:
                epsilon = self.alt_epsilon
                self.epsilon = np.min((1, (self.epsilon + (1-self.eps_dec_factor))))
        else:
            epsilon = self.epsilon
        if not(self.random_is_enough()) or (rand < epsilon):
            # predicted_rewards = np.random.uniform(low=0, high=1, size=4).astype(np.float32)
            # action = np.argmax(predicted_rewards)
            action = np.random.choice(self.disc_actions)
            if self.visualise:
                self.random_action_taken.append(action)
        else:
            predicted_rewards = self.dqn.q_network.forward(torch.tensor(self.state).type(torch.float32))
            action = np.argmax(predicted_rewards.detach().numpy())

        cont_action = np.zeros(2)
        if self.use_experimental_action:
            # x: East - West
            cont_action[0] = predicted_rewards[1] - predicted_rewards[3]
            # y: North - South
            cont_action[1] = predicted_rewards[0] - predicted_rewards[2]
            # Scale
            cont_action = cont_action * (self.step_size / np.linalg.norm(np.abs(cont_action)) )
        else:
            cont_action = self._discrete_action_to_continuous(action)

        self.num_steps_taken += 1

        self.state = state
        self.action = action

        return cont_action

    def _discrete_action_to_continuous(self, discrete_action):
        continuous_action = self.cont_actions[int(discrete_action)]
        return continuous_action

    def random_is_enough(self):
        return (self.dqn.Buffer.is_enough() and (self.num_steps_taken > (self.episode_length * self.num_random_epochs)))

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Convert the distance to a reward

        reward = self.get_reward(distance_to_goal)
        
        # Create a transition
        transition = (self.state, self.action, reward, next_state)
        # Now you can do something with this transition ...
        self.dqn.Buffer.append(transition)
        if self.use_alt_epsilon:
            if np.all(self.state == next_state):
                self.dqn.Buffer.use_alt_epsilon = True
            else:
                self.dqn.Buffer.use_alt_epsilon = False
        
        if self.random_is_enough():
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
        reward = float(0.1*(1 - distance_to_goal))
        if self.reward_type == 'linear':
            return reward
        if self.reward_type == 'exponential':
            if distance_to_goal < 0.25:
                reward *= 3.5
            elif distance_to_goal < 0.5:
                reward *= 2.5
            elif distance_to_goal < 0.75:
                reward *= 1.5
        return reward

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        # Here, the greedy action is fixed, but you should change it so that it returns the action with the highest Q-value
        # action = np.array([0.02, 0.0], dtype=np.float32)
        self.eval_mode = True
        self.log()
        prediction = self.dqn.q_network.forward(torch.tensor(state)).detach()
        action = int(np.argmax(prediction.detach().numpy()))
        action = self._discrete_action_to_continuous(action)
        return action

    def decay_epsilon_strategy(self):
        # dqn.epsilon = np.max((min_eps, dqn.epsilon-(1-eps_dec_factor)))
        self.epsilon = self.epsilon * self.eps_dec_factor

    def log(self):
        if self.eval_mode:
            self.log_q_values()

        if not(self.random_is_enough()):
            return

        if not(self.random_action_taken):
            action_percs = 0
            total_rand = 0
        else:
            action_percs = []
            total_rand = len(self.random_action_taken)
            for act in self.disc_actions:
                action_percs.append((act, (self.random_action_taken.count(act))/self.episode_length))
        print('Random actions were taken {} perc. of the time: {}'.format(round(total_rand/self.episode_length, 4), action_percs))
        self.dqn.losses.append(self.dqn.last_loss)
        self.dqn.icm_losses.append(self.dqn.icm_loss)
        self.log_loss()
        self.log_q_values()

        self.random_action_taken = []

    def log_loss(self):
        # Plot and save the loss vs iterations graph
        variance = np.std(self.dqn.losses)
        iterations = np.arange(0, len(self.dqn.losses))
        self.dqn.losses = [i if i is not None else 0 for i in self.dqn.losses]  # Remove 'None's

        print("Loss: {}".format(self.dqn.losses[-1]))
        if self.dqn.use_prioritisation:
            print("Buffer Alpha: {}".format(self.dqn.Buffer.alpha))

        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.set(xlabel='Iteration', ylabel='Loss', title=('Loss Curve, Batch_size={} '+self.text_eps).format(self.mini_batch_size))
        ax1.plot(iterations, self.dqn.losses, color='blue')
        ax1.text((0.8*len(iterations)), (0.8*np.max(self.dqn.losses)), 'std={}'.format(variance))
        ax2.set(xlabel='Iteration', ylabel='ICM Loss', title=('ICM Loss Curve, Batch_size={} '+self.text_eps).format(self.mini_batch_size))
        ax2.plot(iterations, self.dqn.icm_losses, color='green')
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        fig.savefig("Q2/loss__{}.png".format(self.fig_text))

    def log_q_values(self):

        for i, sx in enumerate(self.disc_states_vector):
            for j, sy in enumerate(self.disc_states_vector):
                prediction = self.dqn.q_network.forward(torch.tensor([sx, sy])).detach().numpy()
                self.q_values[i, j] = np.max(prediction)
                self.v1_values[i, j] = np.min(prediction)
                self.v2_values[i, j] = np.mean(prediction)
                self.v3_values[i, j] = self.q_values[i, j] - self.v1_values[i, j]
        fig, ((ax, ax1), (ax2, ax3)) = plt.subplots(2, 2, sharey=True)
        vmin=np.min((self.q_values, self.v1_values, self.v2_values, self.v3_values))
        vmax=np.max((self.q_values, self.v1_values, self.v2_values, self.v3_values))
        im = ax.imshow(np.transpose(self.q_values), vmin=vmin, vmax=vmax)
        im = ax1.imshow(np.transpose(self.v1_values), vmin=vmin, vmax=vmax)
        im = ax2.imshow(np.transpose(self.v2_values), vmin=vmin, vmax=vmax)
        im = ax3.imshow(np.transpose(self.v3_values), vmin=vmin, vmax=vmax)
        ax.set_title('Max Q values')
        ax1.set_title('Min Q values')
        ax2.set_title('Mean Q values')
        ax3.set_title('Range of Q values')
        fig.colorbar(im, ax=[ax, ax1, ax2, ax3])
        fig.savefig("Q2/QValues__{}.png".format(self.fig_text))


class ReplayBuffer:

    def __init__(self, buffer_alpha_dec=0.999, sample_size=100, maxlen=5000, use_prioritisation=True, use_penalisation=True, use_importance_sampling=True, use_online_learning=False):
        self.buffer = deque(maxlen=maxlen)
        self.sample_size = sample_size
        self.use_alt_epsilon = False
        self.use_importance_sampling = use_importance_sampling
        self.use_online_learning = use_online_learning
        self.use_penalisation = use_penalisation
        self.use_prioritisation = use_prioritisation

        if use_prioritisation:
            self.buffer_probs = deque(maxlen=maxlen)
            self.default_prob = 0.01
            self.min_prob = 0.0001
            self.alpha_decay = buffer_alpha_dec
            self.alpha = 1 if buffer_alpha_dec else 0
            self.recent_idxs = []
        if use_importance_sampling:
            self.beta = 0.4
            self.weights = []

    def is_enough(self):
        if len(self.buffer) > self.sample_size:
            return True
        else:
            return False

    def update_priority(self, weights, sub_idxs=None):
        weights = weights if (np.size(weights)>1) else [weights]
        for i, w in zip(self.recent_idxs, weights):
            # self.buffer_probs[i] = self.min_prob + np.abs(w)
            self.buffer_probs[i] = self.default_prob
        # if sub_idxs is not None:
        #     for sub_i in self.recent_idxs[sub_idxs]:
        #         self.buffer_probs[i] = self.default_prob

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
                if self.alpha > 1e-5:
                    buffer_probs = (np.array(self.buffer_probs) ** self.alpha)
                else:
                    buffer_probs = np.array(self.buffer_probs)
                buffer_probs =  buffer_probs / np.sum(buffer_probs)
                random_indices = np.random.choice(np.arange(buffer_size), size=sample_size, replace=False, p=buffer_probs)

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
                # rew = rew - 0.02
                rew = rew / 1.4
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
        self.output_dimension = output_dimension
        layer_size = 100
        
        self.head_1 = torch.nn.Linear(in_features=input_dimension, out_features=layer_size)
        self.layerA = torch.nn.Linear(in_features=layer_size, out_features=layer_size)
        self.layerV = torch.nn.Linear(in_features=layer_size, out_features=layer_size)
        self.advantage = torch.nn.Linear(in_features=layer_size, out_features=output_dimension)
        self.value = torch.nn.Linear(in_features=layer_size, out_features=1)

    def forward(self, input):
        x = torch.relu(self.head_1(input))
        x_A = torch.relu(self.layerA(x))
        x_V = torch.relu(self.layerV(x))

        value = self.value(x_V)
        value = value.expand(input.size(0), self.output_dimension)
        advantage = self.advantage(x_A)
        Q = value + advantage - advantage.mean()
        return Q


class Forward_Model(torch.nn.Module):
    def __init__(self, input_dimension, output_dimension, encoder_dimension, hidden_size=256):
        super(Forward_Model, self).__init__()
        self.encoder_dimension = encoder_dimension
        self.output_dimension = output_dimension
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(output_dimension+encoder_dimension, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, encoder_dimension))

    def forward(self, state, action):
        ohe_action = torch.zeros(action.shape[0], self.output_dimension)
        indices = torch.stack((torch.arange(action.shape[0]), action.squeeze().long()), dim=0)
        indices = indices.tolist()
        ohe_action[indices] = 1.

        x = torch.cat((state, ohe_action), dim=1)
        x = self.layer1(x)
        return x


class Inverse_Model(torch.nn.Module):
    def __init__(self, input_dimension, output_dimension, hidden_size=256):
        super(Inverse_Model, self).__init__()
        self.input_dimension = input_dimension
        layer_size = 128
        self.encoder = torch.nn.Sequential(torch.nn.Linear(input_dimension, layer_size), torch.nn.ELU())
        self.layer1 = torch.nn.Linear(2*layer_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, output_dimension)
        self.softmax = torch.nn.Softmax(dim=1)

    def calc_forward_dimension(self):
        x = torch.zeros(self.input_dimension).unsqueeze(0)
        x = self.encoder(x)
        forward_dimension = x.flatten().shape[0]
        return forward_dimension

    def forward(self, enc_state, enc_state1):
        x = torch.cat((enc_state, enc_state1), dim=1)
        x = torch.relu(self.layer1(x))
        x = self.softmax(self.layer2(x))
        return x


class ICM(torch.nn.Module):
    def __init__(self, inverse_model, forward_model, learning_rate=1e-3, lambda_=0.1, beta=0.2):
        super(ICM, self).__init__()
        self.inverse_model = inverse_model
        self.forward_model = forward_model
        
        self.forward_scale = 1.
        self.inverse_scale = 1e4
        self.lr = learning_rate
        self.beta = beta
        self.lambda_ = lambda_
        self.forward_loss = torch.nn.MSELoss(reduction='none')
        self.inverse_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.optimiser = torch.optim.Adam(list(self.forward_model.parameters())+list(self.inverse_model.parameters()), lr=1e-3)

    def calc_errors(self, state1, state2, action):
        enc_state1 = self.inverse_model.encoder(state1).view(state1.shape[0],-1)
        enc_state2 = self.inverse_model.encoder(state2).view(state1.shape[0],-1)

        # calc forward error 
        forward_pred = self.forward_model(enc_state1.detach(), action)
        forward_pred_err = 0.5 * self.forward_loss(forward_pred, enc_state2.detach()).sum(dim=1).unsqueeze(dim=1)
        
        # calc prediction error
        pred_action = self.inverse_model(enc_state1, enc_state2) 
        inverse_pred_err = self.inverse_loss(pred_action, action.flatten().long()).unsqueeze(dim=1)    

        return forward_pred_err, inverse_pred_err

    def update_ICM(self, forward_err, inverse_err):
        self.optimiser.zero_grad()
        loss = ((1. - self.beta)*inverse_err + self.beta*forward_err).mean()
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.inverse_model.parameters(),1)
        torch.nn.utils.clip_grad_norm_(self.forward_model.parameters(),1)
        self.optimiser.step()
        return loss.detach().numpy()


class DQN:

    def __init__(self, buffer_alpha_dec, discount, eta, lr, mini_batch_size, network_type, tau, torch_seed, use_curiosity, use_double_q, use_penalisation, use_prioritisation): 
        torch.manual_seed(torch_seed)

        # Architecture
        self.network_type = network_type
        if network_type == 'dueling':
            self.q_network = DuelingNetwork(input_dimension=2, output_dimension=4)
        elif network_type == 'linear':
            self.q_network = Network(input_dimension=2, output_dimension=4)
        self.t_network = copy.deepcopy(self.q_network)
        # self.t_network = Network(input_dimension=2, output_dimension=4)
        
        # Strategies
        self.use_curiosity = use_curiosity
        self.use_nstep_discount = False
        self.use_double_q = use_double_q  # 'target': classic target network | 'action': Use Q net for Q values, T net for action | 'values': Use T net for Q values, Q net for action
        self.loss_type = 'bellman'

        if self.use_curiosity:
            self.eta = eta
            inverse_net = Inverse_Model(input_dimension=2, output_dimension=4)
            forward_net = Forward_Model(input_dimension=2, output_dimension=4, encoder_dimension=inverse_net.calc_forward_dimension())
            self.ICM = ICM(inverse_net, forward_net)

        # HPs
        self.discount = discount
        self.discount_vector = torch.tensor([self.discount ** i for i in range(mini_batch_size)])
        self.lr = lr
        self.tau = tau

        # Memory
        self.use_penalisation = use_penalisation
        self.use_prioritisation = use_prioritisation
        self.use_importance_sampling = False
        maxlen = 10000
        self.Buffer = ReplayBuffer(buffer_alpha_dec, mini_batch_size, maxlen, self.use_prioritisation, self.use_penalisation, self.use_importance_sampling)

        # Technical
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.loss_criterion = torch.nn.MSELoss()

        # Logging
        self.losses = []
        self.last_loss = None
        self.icm_losses = []
        self.icm_loss = None

    def update_t_network(self):
        q_weights = self.q_network.state_dict()
        self.t_network.load_state_dict(q_weights)

    def soft_update_t_network(self):
        for t_param, q_param in zip(self.t_network.parameters(), self.q_network.parameters()):
            t_param.data.copy_(self.tau * q_param.data + (1.0-self.tau)*t_param.data)

    def train_q_network(self, data):

        loss = self._calculate_loss(data)
        self.optimiser.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1)
        self.optimiser.step()

        self.last_loss = loss.item()  # for logging
        return loss.item()

    def _calculate_loss(self, data_batch):
        # Input: S, A
        state, action, reward, state_ = data_batch
        reward = torch.tensor(reward).unsqueeze(1).type(torch.float32)
        action = torch.unsqueeze(torch.tensor(action), 1)

        # Convert to Torch tensor
        state = torch.tensor(state).type(torch.float32)
        state_ = torch.tensor(state_).type(torch.float32)

        # calculate curiosity
        if self.use_curiosity:
            forward_pred_err, inverse_pred_err = self.ICM.calc_errors(state1=state, state2=state_, action=action)
            r_i = self.eta * forward_pred_err
            reward += r_i.detach()
            self.icm_loss = self.ICM.update_ICM(forward_pred_err, inverse_pred_err)

        # Forward
        prediction = self.q_network.forward(state)
        prediction_ = self.t_network.forward(state_).detach()
        
        predicted_action = torch.gather(prediction, -1, action)
        q_prediction_ = self.q_network.forward(state_).detach()
        if self.use_double_q == 'action':
            max_action = torch.unsqueeze(torch.argmax(prediction_, -1), 1)
            predicted_action_ = torch.gather(q_prediction_, -1, max_action).detach()
        elif self.use_double_q == 'values':
            max_action = torch.unsqueeze(torch.argmax(q_prediction_, -1), 1)
            predicted_action_ = torch.gather(prediction_, -1, max_action).detach()
        else:
            # Q(S,A)
            predicted_action_ = prediction_.max(1)[0].detach()  #.numpy()

        if self.use_nstep_discount:
            discount = self.discount_vector[0:len(predicted_action_)]
        else:
            discount = self.discount

        # Bellman
        # --> G = R + γ * max_a [ Q(S', a) ]
        prediction_reward = reward.squeeze() + discount * predicted_action_.squeeze()

        if self.loss_type == 'bellman':
            # --> θ = θ - α * (1/N) * sum[ G - Q(S, A) ]^2
            # loss = self.loss_criterion(torch.squeeze(prediction_reward), torch.squeeze(predicted_action))
            w = ((torch.squeeze(prediction_reward) - torch.squeeze(predicted_action)) ** 2)
            loss = torch.mean(w)
        else:
            loss = self.loss_criterion(reward, predicted_action)

        # Buffer update
        if self.use_prioritisation:
            buffer_idxs = (reward > torch.mean(
                reward, axis=0) + torch.std(reward, axis=0)).squeeze()
            buffer_weights = w.detach().numpy()
            self.Buffer.update_priority(weights=np.array(buffer_weights), sub_idxs=buffer_idxs)
        return loss



def main():
    # TODO penalise wall bumping -> q values exploding
    # TODO fix prioritisation
    # TODO intrinsic motivation
    # TODO dueling heads
    # TODO importance sampling
    hyperparameters = {
        'episode_length': 250,
        'eps decay': 0.999,
        'replay buffer max': 5000,
        'minibatch': 100
    }

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