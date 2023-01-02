import gym
import numpy as np
import pandas as pd
from models import *
from visualization import *

def simulate(environment, policy, num_stimulations=100):
    
    wins = 0
    
    for _ in range(num_stimulations):
        
        done = False
        state = environment.reset()
        while not done:
            
            # pick an action based on our policy table
            action = np.argmax(policy[state])
            
            next_state, reward, done, _ = environment.step(action)
            
            # check if the agent reaches the goal
            # the goal is reached when the game is done and the reward is 1
            if done and reward == 1.0:
                wins += 1
            else:
                state = next_state
    
    return wins

if __name__ == "__main__":

    environment = gym.make('FrozenLake-v1', is_slippery=True)

    num_trials = 50

    outcomes = []
    for trial in range(num_trials):

    
        agentPI = PolicyIteration(environment.env)
        agentVI = ValueIteration(environment.env)
        agentVI.value_iteration()
        agentPI.policy_iteration()

        outcome = [simulate(environment.env, agentPI.policy), simulate(environment.env, agentVI.policy)]

        outcomes.append(outcome)


    df = pd.DataFrame(outcomes, columns=['PI', 'VI'])
    df.to_csv("test.csv", index=False)




    # print("V:")
    # print(agentPI.V)
    # print("policy:")
    # print(agentPI.policy)
    # print_policy(agentPI.policy)
    # show_policy(agentPI.policy, "policy-iteration")
    # show_value_function(agentPI.V, "policy-iteration")
    # print("V:")
    # print(agentVI.V)
    # print("policy:")
    # print(agentVI.policy)
    # print_policy(agentVI.policy)
    # show_policy(agentVI.policy, "value-iteration")
    # show_value_function(agentVI.V, "value-iteration")