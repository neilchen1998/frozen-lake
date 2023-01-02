import numpy as np

class PolicyIteration:

    def __init__(self, env, alpha=0.7, gamma=0.9, theta=1e-8, max_iterations=1e4) -> None:
        
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations

        self.V = 0.01 * np.random.rand(self.env.nS) # the value function
        self.policy = np.ones([self.env.nS, self.env.nA]) / self.env.nA # the policy table

        pass

    def policy_evaluation(self):

        # iterate
        for num_iterations in range(1, int(self.max_iterations)+1):
            
            # this var denotes the change between two iterations
            # if the change is insignificant then we break prematurely
            delta = 0
            
            for state in range(self.env.nS):
                
                # the new value of the current state
                v = 0
                
                # iterate all actions that the agent can take at this state
                for action, action_prob in enumerate(self.policy[state]):
                    
                    # iterate all state that the agent can go
                    for state_prob, next_state, reward, done in self.env.P[state][action]:
                        
                        # calculate the new value of V
                        v += action_prob * state_prob * (reward + self.gamma * self.V[next_state])
                
                # calculate the max. change of this episode
                delta = max(np.abs(self.V[state] - v), delta)
                self.V[state] = v
                                                            
            if (delta < self.theta):
                # print("Policy evaluated in {} iterations".format(num_iterations))
                return
        
        return

    def _next_step(self, state):
    
        action_table = np.zeros(self.env.nA)
        
        for action in range(self.env.nA):
            
            # iterate all state that the agent can go next
            for state_prob, next_state, reward, _ in self.env.P[state][action]:
                
                action_table[action] += state_prob * (reward + self.gamma * self.V[next_state])
                
        return action_table

    def policy_iteration(self):
    
        # repeat the process until the policy converges or the number of iterations is reached
        for num_evaluation in range(int(self.max_iterations)+1):
            
            policy_stable = True
            
            # evaluate the current policy
            V = self.policy_evaluation()
            
            # policy improvement part
            for state in range(self.env.nS):
                
                # get the current action based on current policy
                cur_action = np.argmax(self.policy[state])
                
                action_value = self._next_step(state)
                
                iterated_action = np.argmax(action_value)
                
                # check if our policy table changes or not
                if cur_action != iterated_action:
                    
                    policy_stable = False
                    
                    # update our policy table
                    self.policy[state] += (self.alpha*np.eye(self.env.nA)[iterated_action])
                    row_sums = self.policy[state].sum()
                    self.policy[state] = self.policy[state] / row_sums
            
            # if the policy table becomes stable,
            # then we can return it
            if policy_stable:
                # print("Evaluated at {} iterations".format(num_evaluation))
                return

        return

class ValueIteration:

    def __init__(self, env, alpha=0.7, gamma=0.9, theta=1e-8, max_iterations=1e4) -> None:

        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations

        self.V = 0.01 * np.random.rand(self.env.nS) # the value function
        self.policy = np.zeros([self.env.nS, self.env.nA], dtype=int) # the policy table

        pass

    def value_iteration(self):

        for _ in range(int(self.max_iterations)):

            delta = 0

            for state in range(self.env.nS):

                action_val = self._next_step(state)

                new_val = np.max(action_val)

                delta = max(np.abs(self.V[state] - new_val), delta)

                # update the value
                self.V[state] = new_val

            if delta < self.theta:
                # print("Value table evaluated in {} iterations".format(num_iterations))
                break

        self._create_policy_table()
        
        return

    def _create_policy_table(self):

        for state in range(self.env.nS):

            action_value = self._next_step(state)

            optimal_pi = np.argmax(action_value)

            self.policy[state, optimal_pi] = 1

        return

    def _next_step(self, state):
    
        action_table = np.zeros(self.env.nA)
        
        for action in range(self.env.nA):
            
            # iterate all state that the agent can go next
            for state_prob, next_state, reward, _ in self.env.P[state][action]:
                
                action_table[action] += state_prob * (reward + self.gamma * self.V[next_state])
                
        return action_table