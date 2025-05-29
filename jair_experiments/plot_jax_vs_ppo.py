import matplotlib.pyplot as plt
import numpy as np


def plot_jax_vs_ppo(instance, upto=None):

    # load baseline data
    ppo_data = np.genfromtxt(f'ppo_instance{instance}_mean.csv', delimiter=',')
    jax_data = np.genfromtxt(f'jaxplan_instance{instance}_return_slp.csv')
    if upto is not None:
        ppo_data = ppo_data[:upto]
        jax_data = jax_data[:upto]
    if jax_data.shape[0] < ppo_data.shape[0]:
        padding = np.repeat(
            jax_data[-1:, :], repeats=ppo_data.shape[0] - jax_data.shape[0], axis=0)
        jax_data = np.concat([jax_data, padding], axis=0)
    ppo_mean = np.mean(ppo_data, axis=1)
    ppo_se = np.std(ppo_data, axis=1) / np.sqrt(ppo_data.shape[1])
    jax_mean = np.mean(jax_data, axis=1)
    jax_se = np.std(jax_data, axis=1) / np.sqrt(jax_data.shape[1])
    x = np.arange(ppo_mean.shape[0])

    # plot
    plt.subplot()
    plt.plot(x, jax_mean, color='black', linewidth=2, label='JaxPlan')
    plt.fill_between(x, jax_mean - jax_se, jax_mean + jax_se, color='black', alpha=0.15)
    plt.plot(x, ppo_mean, color='black', linestyle='--', linewidth=1, label='PPO')
    plt.fill_between(x, ppo_mean - ppo_se, ppo_mean + ppo_se, color='black', alpha=0.15)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward per Episode')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'jaxplan_vs_ppo_instance{instance}.pdf')
    plt.clf()


if __name__ == '__main__':
    plot_jax_vs_ppo(1, 5000)
    plot_jax_vs_ppo(3, 5000)
    plot_jax_vs_ppo(10, 10000)