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

    def __init__(self):
        self.buffer = deque(maxlen=5000)
        self.buffer_probs = deque(maxlen=5000)
        self.default_prob = 0.1

    def sample(self, sample_size=20):
        sample = []
        buffer_size = len(self.buffer)
        sample_size = sample_size if buffer_size > sample_size else buffer_size
        
        # Choose 'sample_size' number of random indices based on probabilities in buffer_probs
        total_prob = np.sum(self.buffer_probs)
        if total_prob:
            buffer_probs = np.array(self.buffer_probs) / np.sum(self.buffer_probs)
            random_indices = np.random.choice(list(range(buffer_size)), size=sample_size, replace=False, p=buffer_probs)
        else:
            random_indices = np.random.randint(0, buffer_size, size=sample_size)  # No probs
        for idx in random_indices:
            sample.append(self.buffer[idx])

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
        self.buffer_probs.append(self.default_prob)


# The Agent class allows the agent to interact with the environment.
class Agent:

    # The class initialisation function.
    def __init__(self, environment):
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

    def get_state_as_idx(self, state):
        sx, sy = (state - (self.action_length/2)) / self.action_length
        return [int(sx), int(sy)]

    # Function to reset the environment, and set the agent to its initial state. This should be done at the start of every episode.
    def reset(self):
        # Reset the environment for the start of the new episode, and set the agent's state to the initial state as defined by the environment.
        self.state = self.environment.reset()
        # Set the agent's total reward for this episode to zero.
        self.total_reward = 0.0

    # Function to make the agent take one step in the environment.
    def step(self, q_values=None, action=None):
        # Choose the next action.
        if action:
            discrete_action = action
        else:
            discrete_action = self._choose_next_action(q_values)
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
    def _choose_next_action(self, q_values=None):
        # Return discrete action 0
        if q_values.any():
            sx, sy = self.get_state_as_idx(self.state)
            action = np.argmax(q_values[sx, sy, :], axis=-1)
        else:
            action = np.random.randint(0, self.action_size)
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
    def __init__(self, world_length, agent):
        # Create a Q-network, which predicts the q-value for a particular state.
        # NOTE: Previously I changed input dimension from default 2 to 3, and output dimension from 4 to 1 -> changed back
        self.q_network = Network(input_dimension=2, output_dimension=4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.005)

        self.loss_criterion = torch.nn.MSELoss()

        self.world_length = world_length 
        self.agent = agent

        self.q_values = self.init_q_values()

    def init_q_values(self):
        num_states = int(self.world_length / self.agent.action_length)
        states_vector = np.array(np.arange(0, self.world_length, self.agent.action_length), dtype=np.float32) + (self.agent.action_length / 2)
        q_values = np.zeros((num_states, num_states, self.agent.action_size))
        return q_values

    def update_q_values(self, states, prediction):
        for i, x_state in enumerate(states_vector):
            for j, y_state in enumerate(states_vector):
                in_state = torch.tensor((x_state, y_state))
                q_values[i, j, :] = self.q_network.forward(in_state).detach().numpy()
        return q_values

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, data):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(data)
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
        minibatch_inputs = state  # [state[0], state[1]]
        # Label: R
        minibatch_labels = reward

        # Get Q values from current observations (s, a) using model nextwork
        # Qsa = self.network(states).gather(1, actions)

        # Convert the NumPy array into a Torch tensor
        minibatch_input_tensor = torch.tensor(minibatch_inputs)
        minibatch_labels_tensor = torch.tensor(minibatch_labels)
        # Do a forward pass of the network using the inputs batch
        network_A_prediction = self.q_network.forward(minibatch_input_tensor)
        
        self.update_q_values(state, network_A_prediction)
        # network_prediction = torch.argmax(network_A_prediction)
        # network_prediction = network_A_prediction[action]
        # network_prediction = network_A_prediction[list(range(0,len(action))), action]
        action = torch.tensor(np.array(action).reshape(-1, 1))
        network_prediction = torch.gather(network_A_prediction, 1, action)

        loss = self.loss_criterion(network_prediction, minibatch_labels_tensor)
        return loss


# Main entry point
if __name__ == "__main__":

    # Create an environment.
    # If display is True, then the environment will be displayed after every agent step. This can be set to False to speed up training time. The evaluation in part 2 of the coursework will be done based on the time with display=False.
    # Magnification determines how big the window will be when displaying the environment on your monitor. For desktop monitors, a value of 1000 should be about right. For laptops, a value of 500 should be about right. Note that this value does not affect the underlying state space or the learning, just the visualisation of the environment.
    magnification = 500
    world_length = 1.0
    environment = Environment(display=False, magnification=500)
    # Create an agent
    agent = Agent(environment)
    # Create a DQN (Deep Q-Network)
    dqn = DQN(world_length, agent)
    
    # Create replay buffer
    Buffer = ReplayBuffer()


    # Create lists to store the losses and epochs
    losses = []
    iterations = []
    # Create a graph which will show the loss as a function of the number of training iterations
    fig, ax = plt.subplots()
    ax.set(xlabel='Iteration', ylabel='Loss', title='Loss Curve for Torch Example')

    mini_batch_size = 10
    loss = None
    # Loop over episodes
    total_steps = 1000
    for training_iteration in range(total_steps):
        # Reset the environment for the start of the episode.
        agent.reset()
        # Loop over steps within this episode. The episode length here is 20.
        for step_num in range(20):
            # Step the agent once, and get the transition tuple for this step
            q_values = dqn.get_q_values(world_length, agent)
            transition = agent.step(q_values)
            Buffer.append(transition)
            if len(Buffer.buffer) < mini_batch_size:
                continue
            mini_batch = Buffer.sample(sample_size=mini_batch_size)
            loss = dqn.train_q_network(mini_batch)
            # Sleep, so that you can observe the agent moving. Note: this line should be removed when you want to speed up training
            # time.sleep(0.2)

        # Get the loss as a scalar value
        print('Iteration ' + str(training_iteration) + ', Loss = ' + str(loss))
        # Store this loss in the list
        losses.append(loss)
        iterations.append(training_iteration)
    
    # Visualise Q values
    q_values = dqn.get_q_values(world_length, agent)
    visualiser = QValueVisualiser(environment=environment, magnification=magnification)
    visualiser.draw_q_values(q_values)

    # Draw optimal policy
    policy = np.argmax(q_values, axis=-1)
    optimal_trace = []
    agent.reset()
    for step_num in range(20):
        # Step the agent once, and get the transition tuple for this step
        sx, sy = (agent.state - (agent.action_length/2)) / agent.action_length
        transition = agent.step(action=policy[int(sx), int(sy)])
        optimal_trace.append(transition)
    environment.draw(agent_state=environment.init_state, optimal_trace=optimal_trace)

    # Plot and save the loss vs iterations graph
    ax.plot(iterations, losses, color='blue')
    plt.yscale('log')
    # plt.show()
    fig.savefig("loss_vs_iterations.png")
