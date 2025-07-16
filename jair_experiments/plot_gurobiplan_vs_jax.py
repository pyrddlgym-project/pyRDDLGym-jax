import os
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = '12'
plt.rcParams['axes.labelsize'] = '18'
plt.rcParams['axes.titlesize'] = '18'
plt.rcParams['ytick.labelsize'] = '14'
plt.rcParams['xtick.labelsize'] = '14'
plt.rcParams['legend.fontsize'] = '16'

import numpy as np


def plot_gurobi_vs_jax(instance):

    # load baseline data
    abspath = os.path.dirname(os.path.abspath(__file__))
    grb_data = np.genfromtxt(os.path.join(abspath, f'gurobiplan_instance{instance}.csv'))
    jax_data = np.genfromtxt(os.path.join(abspath, f'gurobiplan_jax_instance{instance}.csv'))
    print(grb_data.shape)
    print(jax_data.shape)
    grb_mean = np.mean(grb_data, axis=1)
    grb_se = np.std(grb_data, axis=1) / np.sqrt(grb_data.shape[1])
    print(f'gurobi result={grb_mean[-1]}')
    jax_mean = np.mean(jax_data, axis=1)
    jax_se = np.std(jax_data, axis=1) / np.sqrt(jax_data.shape[1])
    print(f'jax result={jax_mean[-1]}')
    x = np.arange(grb_mean.shape[0])

    # plot
    plt.subplot()
    plt.plot(x, jax_mean, color='black', linewidth=1.5, label='JaxPlan')
    plt.fill_between(x, jax_mean - jax_se, jax_mean + jax_se, color='black', alpha=0.15)
    plt.plot(x, grb_mean, color='black', linestyle='dotted', linewidth=1, label='GurobiPlan')
    plt.fill_between(x, grb_mean - grb_se, grb_mean + grb_se, color='black', alpha=0.15)
    plt.xlabel('Decision Epoch')
    plt.ylabel('Total Reward per Episode')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'jaxplan_vs_gurobi_instance{instance}.pdf')
    plt.clf()


if __name__ == '__main__':
    plot_gurobi_vs_jax(1)
    plot_gurobi_vs_jax(3)
    plot_gurobi_vs_jax(10)