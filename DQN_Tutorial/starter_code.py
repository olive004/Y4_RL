# Import some modules from other libraries
import numpy as np
import torch
import time
from matplotlib import pyplot as plt
from collections import deque

# Import the environment module
from environment import Environment
from q_value_visualiser import QValueVisualiser


class ReplayBuffer:

    def __init__(self, use_online_learning=False, prioritise_buffer=True):
        self.buffer = deque(maxlen=5000)
        self.use_online_learning = use_online_learning
        self.prioritise_buffer = prioritise_buffer
        if prioritise_buffer:
            self.buffer_probs = deque(maxlen=5000)
            self.default_prob = 0.1
        self.indices_used = []

    def sample(self, sample_size=20):

        if self.use_online_learning:
            state, act, rew, next_state = self.buffer[-1]
            return state, act, rew, next_state

        buffer_size = len(self.buffer)
        sample_size = sample_size if buffer_size > sample_size else buffer_size
        
        # Choose 'sample_size' number of random indices based on probabilities in buffer_probs
        if self.prioritise_buffer:
            total_prob = np.sum(self.buffer_probs)
            if total_prob:
                buffer_probs = np.array(self.buffer_probs) / np.sum(self.buffer_probs)
                random_indices = np.random.choice(list(range(buffer_size)), size=sample_size, replace=False, p=buffer_probs)
        else:
            random_indices = np.random.randint(0, buffer_size, size=sample_size)   # No probs

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

        self.indices_used.append(random_indices)

        return all_s, all_a, all_r, all_next_s

    def append(self, transition):
        self.buffer.append(transition)
        if self.prioritise_buffer:
            self.buffer_probs.append(self.default_prob)


# The Agent class allows the agent to interact with the environment.
class Agent:

    # The class initialisation function.
    def __init__(self, environment, epsilon=0.01, use_rand_actions=False):
        # Set the agent's environment.
        self.environment = environment
        # Create the agent's current state
        self.state = None
        # Create the agent's total reward for the current episode.
        self.total_reward = None
        # Reset the agent.
        self.reset()

        self.action_size = 4
        self.action_length = 0.1

        self.epsilon = epsilon
        self.decay_epsilon = True
        self.use_rand_actions = use_rand_actions

    def get_state_as_idx(self, state):
        world_length = 1.0
        states_vector = np.array(np.arange(0, world_length, self.action_length), dtype=np.float32) + (self.action_length / 2)
        sx, sy = (state - (self.action_length/2)) / self.action_length
        sx, sy = int(round(sx)), int(round(sy))
        # assert round(states_vector[sx], 3) == round(state[0], 3), "Getting wrong states"
        return [int(sx), int(sy)]

    # Function to reset the environment, and set the agent to its initial state. This should be done at the start of every episode.
    def reset(self):
        # Reset the environment for the start of the new episode, and set the agent's state to the initial state as defined by the environment.
        self.state = self.environment.reset()
        # Set the agent's total reward for this episode to zero.
        self.total_reward = 0.0

    # Function to make the agent take one step in the environment.
    def step(self, q_network=None, action=None):
        # Choose the next action.
        if action:
            discrete_action = action
        else:
            discrete_action = self._choose_next_action(q_network)
        # Convert the discrete action into a continuous action.
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        # Take one step in the environment, using this continuous action, based on the agent's current state. This returns the next state, and the new distance to the goal from this new state. It also draws the environment, if display=True was set when creating the environment object..
        next_state, distance_to_goal = self.environment.step(self.state, continuous_action)
        # Compute the reward for this paction.
        reward = self._compute_reward(distance_to_goal)
        # Create a transition tuple for this step.
        transition = (self.state, discrete_action, reward, next_state)
        # Set the agent's state for the next step, as the next state from this step
        self.state = next_state
        # Update the agent's reward for this episode
        self.total_reward += reward
        # Return the transition
        return transition

    # Function for the agent to choose its next action
    def _choose_next_action(self, q_network=None):
        # Choose the intended action
        rand = np.random.rand()
        if self.use_rand_actions or (q_network is None) or (rand < self.epsilon):
            # Random is no Q given
            action = np.random.randint(0, self.action_size)
        else:
            # sx, sy = self.get_state_as_idx(self.state)
            # If all Qs are the same
            predicted_rewards = q_network.forward(torch.tensor(self.state))
            action = np.argmax(predicted_rewards.detach().numpy())
            # if np.all(q_values[sx, sy, :] == q_values[sx, sy, 0]):
            #     action = np.random.randint(0, self.action_size)
            # else:
            #     action = np.argmax(q_values[sx, sy, :], axis=-1)
        return action

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:
            # Move 0.1 to the right, and 0 upwards
            continuous_action = np.array([self.action_length, 0], dtype=np.float32)
        if discrete_action == 1:
            # Move 0.1 to the left, and 0 upwards
            continuous_action = np.array([-self.action_length, 0], dtype=np.float32)
        if discrete_action == 2:
            # Move 0.1 to the up
            continuous_action = np.array([0, self.action_length], dtype=np.float32)
        if discrete_action == 3:
            # Move 0.1 to the down
            continuous_action = np.array([0, -self.action_length], dtype=np.float32)
        return continuous_action

    # Function for the agent to compute its reward. In this example, the reward is based on the agent's distance to the goal after the agent takes an action.
    def _compute_reward(self, distance_to_goal):
        reward = float(0.1*(1 - distance_to_goal))
        return reward


# The Network class inherits the torch.nn.Module class, which represents a neural network.
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


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self, agent, epsilon=0.01, use_bellman_loss=True, use_target=False, world_length=1.0, lr=0.001, discount=0.9, use_online_learning=False, prioritise_buffer=True, ):
        # Create a Q-network, which predicts the q-value for a particular state.
        # NOTE: Previously I changed input dimension from default 2 to 3, and output dimension from 4 to 1 -> changed back
        self.q_network = Network(input_dimension=2, output_dimension=4)
        if use_target:
            self.t_network = Network(input_dimension=2, output_dimension=4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        # Create replay buffer
        self.Buffer = ReplayBuffer(use_online_learning, prioritise_buffer)
        self.use_bellman_loss = use_bellman_loss
        self.use_target = use_target

        # HPs
        self.loss_criterion = torch.nn.MSELoss()
        self.discount = discount
        self.epsilon = epsilon

        # World
        self.world_length = world_length 
        self.agent = agent
        self.states_vector = np.array(np.arange(0, self.world_length, self.agent.action_length), dtype=np.float32) + (self.agent.action_length / 2)

        # Collect along the way
        self.q_values = self.init_q_values()
        self.optimal_network = None

    def init_q_values(self):
        num_states = int(self.world_length / self.agent.action_length)
        q_values = np.zeros((num_states, num_states, self.agent.action_size))
        return q_values

    def update_q_values(self, states=None, prediction=None):
        # for i, state in enumerate(states):
        #     if self.Buffer.use_online_learning:
        #         state = states
        #     sx, sy = self.agent.get_state_as_idx(state)
        #     self.q_values[sx, sy, :] = prediction[i].detach().numpy()

        for i, sx in enumerate(self.states_vector):
            for j, sy in enumerate(self.states_vector):
                state = torch.tensor([sx, sy]).type(torch.float32)
                pred = self.q_network.forward(state)
                self.q_values[i, j, :] = pred.detach().numpy()

    def final_q_value_update(self):
        # Convert NaN's to zero
        self.q_values[np.isnan(self.q_values)] = 0
    
    def get_e_greedy_policy(self):
        # Return epsilon-greedy policy
        num_states = int(self.world_length / self.agent.action_length)
        policy = np.zeros((num_states, num_states, self.agent.action_size))

        optimal_value = 1 - self.epsilon + (self.epsilon/self.agent.action_size)
        suboptimal_value = self.epsilon / self.agent.action_size
        for sx in range(num_states):
            for sy in range(num_states):
                # If this state has the same Q for each action, don't update policy TODO: ASSUMPTION
                if np.all(self.q_values[sx, sy, :] == self.q_values[sx, sy, 0]):
                    policy[sx, sy, :] = 1 / self.agent.action_size
                    continue

                a_optimal = np.argmax(self.q_values[sx, sy, :], axis=-1)  # One optimal action
                # a_optimal = np.where(Q[state, :] == np.max(Q[state, :]))  # Multiple optimal actionss
                # if type(a_optimal) == tuple:
                #     a_optimal = a_optimal[0]

                for a in range(self.agent.action_size):
                    if a == a_optimal:
                        policy[sx, sy, a] = optimal_value
                    else:
                        policy[sx, sy, a] = suboptimal_value
        return policy

    def update_t_network(self):
        q_weights = self.q_network.state_dict()
        self.t_network.load_state_dict(q_weights)

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, data):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition. NOTE: added helper outputs for Q value
        loss = self._calculate_loss(data)
        # self.update_q_values(state, prediction_state)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, mini_batch):
        # pass
        # TODO
        # Input: S, A
        state, action, reward, next_state = mini_batch
        reward = torch.tensor(reward).type(torch.float32)

        # Convert the NumPy array into a Torch tensor
        state_tensor = torch.tensor(state)
        next_state_tensor = torch.tensor(next_state)

        # Do a forward pass of the network using the inputs batch
        # Predict S
        prediction = self.q_network.forward(state_tensor)
        # Predict S'
        if use_target:
            prediction_next = self.t_network.forward(next_state_tensor).detach()
        else:
            prediction_next = self.q_network.forward(next_state_tensor).detach()
        
        # State Q(S,A)
        action = torch.tensor(np.array(action))
        # Next state Q(S',A)
        max_actions = torch.argmax(prediction_next, axis=-1)
        if self.Buffer.use_online_learning:
            prediction_A = prediction[int(action)]
            prediction_next_A = prediction_next.max()
        else:
            action = action.reshape(-1,1)
            reward = reward.reshape(-1,1)  # For making dimensions same
            prediction_A = torch.gather(prediction, -1, action)
        
            max_actions = max_actions.reshape(-1,1)
            # prediction_next_A = torch.gather(prediction_next, 1, max_actions).detach()
            prediction_next_A = prediction_next.max(1)[0].detach().numpy()

        # Bellman
        # --> R_Q = R + γ * max_a [ Q(S', a) ]
        prediction_reward = reward.reshape(1,-1) + self.discount * prediction_next_A
        # prediction_reward = torch.tensor(prediction_reward).type(torch.float32)

        if self.use_bellman_loss:
            # NOTE: Bellman
            # --> θ = θ - α * (1/N) * sum[ R_Q - Q(S, A) ]^2
            loss = self.loss_criterion(torch.squeeze(prediction_reward), torch.squeeze(prediction_A))
            # loss = torch.sum((prediction_A - prediction_reward) ** 2)
        else:
            # NOTE: Part 1: Predicting immediate reward
            loss = self.loss_criterion(reward, prediction_A)
        return loss


# Main entry point
if __name__ == "__main__":
    # TODO general:
    # decay eps
    # e greedy: choose action based on policy

    # Hyperparameters
    magnification = 500
    use_online_learning = False
    mini_batch_size = 1 if use_online_learning else 100
    epsilon = 1
    min_eps = 0.1
    eps_dec_factor = 0.995
    lr = 0.002
    discount = 0.9
    decay_epsilon = True
    use_bellman_loss = True
    use_rand_actions = False
    use_target = False
    
    visualise_last = True
    total_iters = 600
    total_steps = 90
    N_update_target = 10

    # Q 1.1 and 1.2:
    # 1.1a)
    # use_rand_actions, use_online_learning, total_iters, use_target, lr, use_bellman_loss = (True, True, 100, False, 0.001, False)
    # 1.1b)
    # use_rand_actions, use_online_learning, total_iters, use_target, lr, use_bellman_loss = (True, False, 100, False, 0.001, False)

    # # Q 1.3:
    # # 1.3a) 
    # use_rand_actions, use_online_learning, total_iters, use_target, lr, N_update_target, use_bellman_loss, epsilon = (True, False, 100, False, 0.001, 10, True, 0)
    # # 1.3b)
    # use_rand_actions, use_online_learning, total_iters, use_target, lr, N_update_target, use_bellman_loss, epsilon = (True, False, 100, True, 0.001, 10, True, 0)

    # # Q 1.4
    use_rand_actions, use_online_learning, total_iters, use_target, lr, N_update_target, use_bellman_loss, epsilon = (False, False, 600, True, 0.002, 10, True, 1)
    
    mini_batch_size = 1 if use_online_learning else 100
    text_target = 'tnet' if use_target else 'qnet'
    text_loss_type = 'bllmn' if use_bellman_loss else 'immdrew'
    
    np.random.seed(0)
    torch.manual_seed(6)
    torch.set_default_dtype(torch.float32)

    # Create an environment.
    # If display is True, then the environment will be displayed after every agent step. This can be set to False to speed up training time. The evaluation in part 2 of the coursework will be done based on the time with display=False.
    # Magnification determines how big the window will be when displaying the environment on your monitor. For desktop monitors, a value of 1000 should be about right. For laptops, a value of 500 should be about right. Note that this value does not affect the underlying state space or the learning, just the visualisation of the environment.
    environment = Environment(display=False, magnification=500)
    # Create an agent
    agent = Agent(environment, epsilon, use_rand_actions)
    # Create a DQN (Deep Q-Network)
    dqn = DQN(agent, epsilon, use_bellman_loss, use_target, lr=lr, discount=discount, use_online_learning=use_online_learning)
    
    # Create lists to store the losses and epochs
    losses = []
    iterations = []
    biggest_reward = 0
    fig_buff, ax_buff = plt.subplots()
    
    loss = None
    # Loop over episodes
    for training_iteration in range(total_iters):
        # print("Current epsilon = {}".format(dqn.epsilon))

        # Reset the environment for the start of the episode.
        agent.reset()
        # Loop over steps within this episode. The episode length here is 20.
        for step_num in range(total_steps):
            # Step the agent once, and get the transition tuple for this step
            transition = agent.step(dqn.q_network)
            dqn.Buffer.append(transition)
            if len(dqn.Buffer.buffer) < mini_batch_size:  # just for beginning
                continue
            mini_batch = dqn.Buffer.sample(sample_size=mini_batch_size)

            # buffer_indices = np.ones(len(mini_batch)) * training_iteration
            # ax_buff.plot(buffer_indices, dqn.Buffer.indices_used[-1])
            
            # NOTE: TEST
            if use_online_learning:
                assert transition[0].all() == mini_batch[0].all(), "Sampling wrong"

            loss = dqn.train_q_network(mini_batch)
            # Sleep, so that you can observe the agent moving. Note: this line should be removed when you want to speed up training
            # time.sleep(0.2)

            # Update epsilon
            if decay_epsilon:  # and ((training_iteration / total_iters) < 0.1):
                # dqn.epsilon = np.max((min_eps, dqn.epsilon-(1-eps_dec_factor)))
                dqn.epsilon = dqn.epsilon * eps_dec_factor
                agent.epsilon = dqn.epsilon

            # Save if goal was reached or max reward
            if (agent.total_reward > biggest_reward) or np.all(environment.goal_state == transition[0]):
                biggest_reward = np.max((agent.total_reward, biggest_reward))
                dqn.optimal_network = dqn.q_network
                print('Bigger reward or goal reached')

        # Update Target
        if use_target and (training_iteration % N_update_target == 0):
            dqn.update_t_network()

        # Get the loss as a scalar value
        print('Iteration ' + str(training_iteration) + ', Loss = ' + str(loss))
        # Store this loss in the list
        losses.append(loss)
        iterations.append(training_iteration)
    
    # Visualise Q values
    # dqn.final_q_value_update()
    visualiser = QValueVisualiser(environment=environment, magnification=magnification)
    dqn.update_q_values()
    visualiser.draw_q_values(dqn.q_values)

    # Draw optimal GREEDY policy
    # policy = np.argmax(dqn.q_values, axis=-1)
    optimal_trace = []
    agent.reset()
    dqn.epsilon = 0
    for step_num in range(20):
        # Step the agent once, and get the transition tuple for this step
        if not(visualise_last) and dqn.optimal_network:
            transition = agent.step(dqn.optimal_network)
        else:
            transition = agent.step(dqn.q_network)
        optimal_trace.append(transition)
    opt_filename = "loss_vs_iterations_batch{}_lr{}_eps{}_steps{}_{}_{}.png".format(mini_batch_size, lr, epsilon, total_iters, text_loss_type, text_target)
    environment.draw(agent_state=environment.init_state, optimal_trace=optimal_trace, opt_filename=opt_filename)

    # Plot and save the loss vs iterations graph
    fig, ax = plt.subplots()
    text_eps = 'Epsilon decaying' if decay_epsilon else ''
    losses = [i if i is not None else 0 for i in losses]  # Remove 'None's
    variance = np.std(losses)
    ax.set(xlabel='Iteration', ylabel='Loss', title=('Loss Curve, Batch_size={} '+text_eps).format(mini_batch_size))
    ax.plot(iterations, losses, color='blue')
    plt.text(iterations[int(0.8*len(iterations))], losses[int(0.8*np.max(losses))], 'std={}'.format(variance))
    plt.yscale('log')
    # plt.show()
    fig.savefig("loss_vs_iterations_batch{}_lr{}_eps{}_steps{}_{}_{}.png".format(mini_batch_size, lr, epsilon, total_iters, text_loss_type, text_target))

    # fig, ax = plt.subplots()
    # if not(use_online_learning):
    #     for i in iterations:
    #         for s in range(total_steps)
    #             # buffer_indices.append(list(zip((np.ones(len(dqn.Buffer.indices_used[i])) * i), dqn.Buffer.indices_used[i])))
    #             buffer_indices = np.ones(len(mini_batch)) * training_iteration
    #             ax.plot(buffer_indices, dqn.Buffer.indices_used[i])
    # ax.set(xlabel='Iteration', ylabel='Buffer index', title=('Buffer indices sampled, Batch_size={} '+text_eps).format(mini_batch_size))
    # fig.savefig("loss_vs_iterations_batch{}_lr{}_eps{}_steps{}.png".format(mini_batch_size, lr, epsilon, total_iters))

    # Save Numpy 
    np.save('losses_{}_batch{}_lr{}_eps{}_steps{}.npy'.format(text_loss_type, mini_batch_size, lr, epsilon, total_iters), losses)

    print(("File saved as " + "loss_vs_iterations_{}_batch{}_lr{}_eps{}_steps{}.png".format(text_loss_type, mini_batch_size, lr, epsilon, total_iters)))
