import time
import numpy as np
import torch

from random_environment_JOE import Environment
from agent import Agent

import optuna

        
# DQN
# self.mini_batch_size = 1 if self.use_online_learning else 100


def sample_dqn_params(trial: optuna.Trial):
    """Sampler for A2C hyperparameters."""
    # Network
    torch_seed = trial.suggest_int("torch_seed", 0, 4)

    # Learning
    use_double_q = trial.suggest_categorical("use_double_q", ["values", "target"])
    lr = trial.suggest_float("lr", 1e-5, 1, log=True)
    discount = 1.0 - trial.suggest_float("discount", 1e-4, 0.1, log=True)
    tau = trial.suggest_float("tau", 0.0001, 1)

    # Rewards & Memory
    use_penalisation = trial.suggest_categorical("use_penalisation", [True, False])
    use_prioritisation = trial.suggest_categorical("use_prioritisation", [True, False])
    min_eps = trial.suggest_categorical("min_eps", [0.01, 0.09, 0.15])
    use_softupdate = trial.suggest_categorical("use_softupdate", [True, False])
    reward_type = trial.suggest_categorical("reward_type", ["linear", "exponential"])

    # Agent
    stop_at_min_eps = trial.suggest_categorical("stop_at_min_eps", [True, False])
    eps_dec_factor = trial.suggest_categorical("eps_dec_factor", [0.99, 0.999])
    N_update_target = trial.suggest_int("N_update_target", 4, 20)
    mini_batch_size = trial.suggest_categorical("mini_batch_size", [32, 64, 100])


    # Display true values
    # trial.set_user_attr("gamma_", gamma)
    # trial.set_user_attr("gae_lambda_", gae_lambda)
    # trial.set_user_attr("n_steps", n_steps)

    return {
        "discount": discount,
        "eps_dec_factor": eps_dec_factor,
        "lr": lr,
        "min_eps": min_eps,
        "mini_batch_size": mini_batch_size,
        "N_update_target": N_update_target,
        "reward_type": reward_type,
        "stop_at_min_eps": stop_at_min_eps,
        "tau": tau,
        "torch_seed": torch_seed,
        "use_double_q": use_double_q,
        "use_penalisation": use_penalisation,
        "use_prioritisation": use_prioritisation,
        "use_softupdate": use_softupdate,
    }



if __name__ == "__main__":

    # This determines whether the environment will be displayed on each each step.
    # When we train your code for the 10 minute period, we will not display the environment.
    display_on = True

    # Create a random seed, which will define the environment
    random_seed = int(time.time())
    np.random.seed(random_seed)

    np.random.seed(3)  # 1606064318

    # Create a random environment
    environment = Environment(magnification=500, difficulty=3)

    # Create an agent
    # kwargs = sample_dqn_params(trial)
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
        if display_on and total_time > 0:
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
        environment.show(state, all_states, 'every')
        state = next_state

    # Print out the result
    if has_reached_goal:
        print('Reached goal in ' + str(step_num) + ' steps.')
    else:
        print('Did not reach goal. Final distance = ' + str(distance_to_goal))



# Main entry point
def objective(trial: optuna.Trial) -> float:
    # Set pytorch num threads to 1 for faster training
    torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    try:
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS, timeout=600)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))