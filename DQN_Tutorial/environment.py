import numpy as np
import cv2


# The Environment class defines the "world" within which the agent is acting
class Environment:

    # Function to initialise an Environment object
    def __init__(self, display, magnification=500):
        # Set whether the environment should be displayed after every step
        self.display = display
        # Set the magnification factor of the display
        self.magnification = magnification
        # Set the initial state of the agent
        self.init_state = np.array([0.35, 0.15], dtype=np.float32)
        # Set the initial state of the goal
        self.goal_state = np.array([0.35, 0.85], dtype=np.float32)
        # Set the space which the obstacle occupies
        self.obstacle_space = np.array([[0.0, 0.7], [0.4, 0.5]], dtype=np.float32)
        # Create an image which will be used to display the environment
        self.image = np.zeros([int(self.magnification), int(self.magnification), 3], dtype=np.uint8)

    # Function to reset the environment, which is done at the start of each episode
    def reset(self):
        return self.init_state

    # Function to execute an agent's step within this environment, returning the next state and the distance to the goal
    def step(self, state, action):
        # Determine what the new state would be if the agent could move there
        next_state = state + action
        # If this state is outside the environment's perimeters, then the agent stays still
        if next_state[0] < 0.0 or next_state[0] > 1.0 or next_state[1] < 0.0 or next_state[1] > 1.0:
            next_state = state
        # If this state is inside the obstacle, then the agent stays still
        if self.obstacle_space[0, 0] <= next_state[0] < self.obstacle_space[0, 1] and self.obstacle_space[1, 0] <= next_state[1] < self.obstacle_space[1, 1]:
            next_state = state
        # Compute the distance to the goal
        distance_to_goal = np.linalg.norm(next_state - self.goal_state)
        # Draw and show the environment, if required
        if self.display:
            self.draw(next_state)
        # Return the next state and the distance to the goal
        return next_state, distance_to_goal

    # Function to draw the environment and display it on the screen, if required
    def draw(self, agent_state, optimal_trace=None, opt_filename=False):
        # Create the background image
        window_top_left = (0, 0)
        window_bottom_right = (self.magnification * 1, self.magnification * 1)
        cv2.rectangle(self.image, window_top_left, window_bottom_right, (246, 238, 229), thickness=cv2.FILLED)
        # Draw the obstacle
        obstacle_left = int(self.magnification * self.obstacle_space[0, 0])
        obstacle_top = int(self.magnification * (1 - self.obstacle_space[1, 1]))
        obstacle_width = int(self.magnification * (self.obstacle_space[0, 1] - self.obstacle_space[0, 0]))
        obstacle_height = int(self.magnification * (self.obstacle_space[1, 1] - self.obstacle_space[1, 0]))
        obstacle_top_left = (obstacle_left, obstacle_top)
        obstacle_bottom_right = (obstacle_left + obstacle_width, obstacle_top + obstacle_height)
        cv2.rectangle(self.image, obstacle_top_left, obstacle_bottom_right, (0, 0, 150), thickness=cv2.FILLED)
        # Create the border
        border_top_left = (0, 0)
        border_bottom_right = (self.magnification * 1, self.magnification * 1)
        cv2.rectangle(self.image, border_top_left, border_bottom_right, (0, 0, 0), thickness=int(self.magnification * 0.02))
        # Draw the goal
        goal_centre = (int(self.goal_state[0] * self.magnification), int((1 - self.goal_state[1]) * self.magnification))
        goal_radius = int(0.02 * self.magnification)
        goal_colour = (227, 158, 71)
        cv2.circle(self.image, goal_centre, goal_radius, goal_colour, cv2.FILLED)
        # Draw the agent
        agent_centre = (int(agent_state[0] * self.magnification), int((1 - agent_state[1]) * self.magnification))
        agent_radius = int(0.02 * self.magnification)
        agent_colour = (100, 199, 246)
        cv2.circle(self.image, agent_centre, agent_radius, agent_colour, cv2.FILLED)
        # Draw optimal policy line
        if optimal_trace:
            for step in optimal_trace:
                state, action, rew, state_ = step
                state = state * self.magnification
                state_ = state_ * self.magnification
                cv2.line(self.image, (int(state[0]), int(self.magnification-state[1])), (int(state_[0]), int(self.magnification-state_[1])), (0,255,0), 6)
            filename = opt_filename if opt_filename else 'optimal_policy_path.png'
            cv2.imshow("Environment", self.image)
            cv2.waitKey(1)
            cv2.imwrite(filename, self.image)

        # Show the image
        cv2.imshow("Environment", self.image)
        # This line is necessary to give time for the image to be rendered on the screen
        cv2.waitKey(1)

