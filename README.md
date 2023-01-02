# Frozen Lake Project

## Overview

The agent wants to cross the frozen lake from Start (S) to Goal (G) without falling into any Holes (H) by walking over the Frozen (F) lake. Since the lake is frozen, the agent may not always move to the grid that it intends[1].

## Approaches

We identify this problem as a stochastic problem and we decide to use two approaches to solve this task: Policy Iteration (PI) and Value Iteration (VI).

* Policy Iteration (PI): Firstly, we randomly initialize a value function. Secondly, we iterate the value function over and over until the it converges (policy evaluation). Thridly, we use that value function to get our policy (policy improvement). 

* Value Iteration (VI): Firstly, we randomly initialize a value function. Secondly, we find the optimal value function. Thirdly, we extract a policy based on that value function.

Our code is based on the following pseudocode[2].

<figure>
  <img src="https://github.com/neilchen1998/frozen-lake/blob/main/graphs/policy-iteration-pseudocode" alt="my alt text" width="300" height="250"/>
  <figcaption align="bottom">Policy iteration</figcaption>
</figure>

&nbsp;

<figure>
  <img src="https://github.com/neilchen1998/frozen-lake/blob/main/graphs/value-iteration-pseudocode" alt="my alt text" width="300" height="250"/>
  <figcaption align="bottom">Value iteration</figcaption>
</figure>

&nbsp;

## Results

<figure>
  <img src="https://github.com/neilchen1998/ai-snake/blob/main/gifs/training-early-stage.gif" alt="my alt text" width="300" height="250"/>
  <figcaption align="bottom">The early stage of training</figcaption>
</figure>

&nbsp;

<figure>
  <img src="https://github.com/neilchen1998/ai-snake/blob/main/gifs/result-graph.png" alt="my alt text" width="300" height="250"/>
  <figcaption align="bottom">A snapshot of the result graph</figcaption>
</figure>

&nbsp;

We run 50 trials, each trial we calculate the value function and the policy, and we run the agent using that information for 100 episodes and sum up the number of times it reaches the goal without falling into one of the holes in the map.

<table>
<caption align="center">Results</caption>
<tr>
    <th></th>
    <th>Policy Iteration</th>
    <th>Value Iteration</th>
</tr>
<tr>
    <td>mean</td>
    <td>78.46</td>
    <td>78.94</td>
</tr>
<tr>
    <td>std</td>
    <td>3.45354</td>
    <td>3.683443</td>
</tr>
<tr>
    <td>min</td>
    <td>70</td>
    <td>67</td>
</tr>
<tr>
    <td>max</td>
    <td>86</td>
    <td>88</td>
</tr>
</table>

## References

1. [Frozen Lake](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/)

2. [Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)