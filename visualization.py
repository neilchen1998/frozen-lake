import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def print_policy(pi):

    """
    Print the policy in a more readable table 
    """

    action2arrow = {0:'<', 1:'v', 2:'>', 3:'^'}
    policy_arrows = np.array([action2arrow[np.argmax(action)] for action in pi])
    print(np.array(policy_arrows).reshape([-1, 4]))

    pass

def show_policy(policy, title):

    """
    Show the policy (heatmap) using seaborn
    """

    df = pd.DataFrame(policy, columns=['left', 'down', 'right', 'up'])

    ax = plt.axes()
    sns.heatmap(df, cmap=sns.color_palette("YlOrBr", as_cmap=True), annot=True, fmt=".2f", ax = ax)

    ax.set_title('Policy')
    ax.set(xlabel="Actions", ylabel="Grids")
    plt.savefig("{}-policy.png".format(title))
    plt.show()

    pass

def show_value_function(V, title):

    """
    Show the value function (heatmap) using seaborn
    """

    V = V.reshape((-1, 4))

    df = pd.DataFrame(V, columns=['0', '1', '2', '3'])

    ax = plt.axes()
    sns.heatmap(df, cmap=sns.color_palette("light:b", as_cmap=True), annot=True, fmt=".3f", ax = ax)

    ax.set_title('Value Function')
    ax.set(xlabel="", ylabel="")
    plt.savefig("{}-value-function.png".format(title))
    plt.show()

    pass