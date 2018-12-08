import numpy as np
from collections import defaultdict


class Agent:
    def __init__(self, nA=6, alpha=0.1, gamma=0.65, epsilon=1):

        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon

        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

        self.method = "ES"
        self.methods = {"ES", "SM", "S"}

    @staticmethod
    def update_Q(Qsa, Qsa_next, reward, alpha, gamma):
        """ updates the action-value function estimate using the most recent time step """
        return Qsa + (alpha * (reward + (gamma * Qsa_next) - Qsa))

    def epsilon_greedy_probs(self, Q_s, epsilon: float):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        policy_s = np.ones(self.nA) * epsilon / self.nA
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / self.nA)
        return policy_s

    def select_action(self, state, epsilon=1.0):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        self.updated_epsilon(epsilon)
        policy_s = self.epsilon_greedy_probs(self.Q[state], epsilon)
        action = np.random.choice(np.arange(self.nA), p=policy_s)
        return action

    def updated_epsilon(self, epsilon):
        """update epsilon base on select_action"""
        self.epsilon = epsilon

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        policy_s = self.epsilon_greedy_probs(self.Q[next_state], epsilon=self.epsilon)

        # Expected Sarsa
        if self.method == "ES":
            self.Q[state][action] = self.update_Q(
                self.Q[state][action],
                np.dot(self.Q[next_state], policy_s),
                reward,
                self.alpha,
                self.gamma,
            )

        # Sarsa max
        elif self.method == "SM":
            self.Q[state][action] = self.update_Q(
                self.Q[state][action],
                np.max(self.Q[next_state]),
                reward,
                self.alpha,
                self.gamma,
            )

        # Sarsa
        elif self.method == "S":
            next_action = self.select_action(state)
            self.Q[state][action] = self.update_Q(
                self.Q[state][action],
                self.Q[next_state][next_action],
                reward,
                self.alpha,
                self.gamma,
            )
        else:
            raise ValueError("method should be in {'ES', 'SM', 'S'}")
