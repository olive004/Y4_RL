import time
import numpy as np
import torch

from random_environment_JOE import Environment
from agent import Agent


# Main entry point
if __name__ == "__main__":

    # This determines whether the environment will be displayed on each each step.
    # When we train your code for the 10 minute period, we will not display the environment.
    display_on = True

    # Create a random seed, which will define the environment
    random_seed = int(time.time())
    np.random.seed(random_seed)

    np.random.seed(4)  # 1606064318
    torch.manual_seed(0)

    # Create a random environment
    environment = Environment(magnification=500, difficulty=2)

    # Create an agent
    agent = Agent()

    # Get the initial state
    state = environment.init_state

    # Determine the time at which training will stop, i.e. in 10 minutes (600 seconds) time
    start_time = time.time()
    end_time = start_time + 600

    # Train the agent, until the time is up
    episode_count = 0
    episode_time = time.time()
    all_states = []  # TODO DELETE
    while time.time() < end_time:
        # If the action is to start a new episode, then reset the state
        if agent.has_finished_episode():
            state = environment.init_state
            all_states = []  # TODO DELETE
            
            print('Time elapsed in episode: {}'.format(time.time() - episode_time))
            total_time = episode_time - start_time
            print('Total time elapsed: {}'.format(total_time))
            episode_time = time.time()
            episode_count += 1
        all_states.append(state)  # TODO DELETE
        # Get the state and action from the agent
        action = agent.get_next_action(state)
        # Get the next state and the distance to the goal
        next_state, distance_to_goal = environment.step(state, action)
        # Return this to the agent
        agent.set_next_state_and_distance(next_state, distance_to_goal)
        # Set what the new state is
        state = next_state
        # Optionally, show the environment
        if display_on and total_time > 40:
            environment.show(state, all_states)

    # Test the agent for 100 steps, using its greedy policy
    state = environment.init_state
    has_reached_goal = False
    all_states = []
    for step_num in range(100):
        action = agent.get_greedy_action(state)
        next_state, distance_to_goal = environment.step(state, action)
        # The agent must achieve a maximum distance of 0.03 for use to consider it "reaching the goal"
        if distance_to_goal < 0.03:
            has_reached_goal = True
            break
        all_states.append(state)
        environment.show(state, all_states)
        state = next_state

    # Print out the result
    if has_reached_goal:
        print('Reached goal in ' + str(step_num) + ' steps.')
    else:
        print('Did not reach goal. Final distance = ' + str(distance_to_goal))
