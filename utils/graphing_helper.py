import matplotlib.pyplot as plt
import numpy as np


# Graphing Functions
def make_rollout_graphs(xs, ys, thetas, a1s, a2s, steps, path):

    # plotting
    fig1 = plt.figure(1)
    fig1.suptitle('Policy Rollout X, Y, Theta vs Time')
    ax1 = fig1.add_subplot(311)
    ax2 = fig1.add_subplot(312)
    ax3 = fig1.add_subplot(313)

    fig2 = plt.figure(2)
    fig2.suptitle('Policy Rollout a1 vs a2')
    ax4 = fig2.add_subplot(111)

    fig3 = plt.figure(3)
    fig3.suptitle('Policy Rollout a1 and a2 vs Time')
    ax5 = fig3.add_subplot(211)
    ax6 = fig3.add_subplot(212)

    fig4 = plt.figure(4)
    fig4.suptitle('Policy Rollout X vs Y')
    ax7 = fig4.add_subplot(111)

    ax1.plot(steps, xs, '.-')
    ax1.set_ylabel('x')
    ax1.set_xlabel('steps')
    ax2.plot(steps, ys, '.-')
    ax2.set_ylabel('y')
    ax2.set_xlabel('steps')
    ax3.plot(steps, thetas, '.-')
    ax3.set_ylabel('theta')
    ax3.set_xlabel('steps')

    ax4.plot(a1s,a2s,'.-')
    ax4.set_xlabel('a1')
    ax4.set_ylabel('a2')

    ax5.plot(steps, a1s, '.-')
    ax5.set_xlabel('a1')
    ax5.set_ylabel('steps')
    ax6.plot(steps, a2s, '.-')
    ax6.set_xlabel('a2')
    ax6.set_ylabel('steps')

    ax7.plot(xs, ys,'.-')
    ax7.set_xlabel('x')
    ax7.set_ylabel('y')

    ax1.grid(True, which='both', alpha=.2)
    ax2.grid(True, which='both', alpha=.2)
    ax3.grid(True, which='both', alpha=.2)
    ax4.grid(True, which='both', alpha=.2)
    ax5.grid(True, which='both', alpha=.2)
    ax6.grid(True, which='both', alpha=.2)
    ax7.grid(True, which='both', alpha=.2)

    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig1.savefig(path + '/Policy_Rollout_X_Y_Theta_vs_Time.png')
    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2.savefig(path + '/Policy_Rollout_a1_vs_a2.png')
    fig3.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig3.savefig(path + '/Policy_Rollout_a1_and_a2_vs_Time.png')
    fig4.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig4.savefig(path + '/Policy_Rollout_X_and_Y.png')

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)


def make_loss_plot(num_episodes, avg_losses, std_losses, path):
    avg_losses = np.array(avg_losses)
    std_losses = np.array(std_losses)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Average Loss vs Number of Iterations')
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Average Loss')
    ax.grid(True, which='both', alpha=.2)
    ax.plot(num_episodes, avg_losses)
    ax.fill_between(num_episodes, avg_losses-std_losses, avg_losses+std_losses, alpha=.2)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(path + '/Average_Loss_vs_Number_of_Iterations.png')
    plt.close()


def make_learning_plot(num_episodes, avg_rewards, std_rewards, path):
    avg_rewards = np.array(avg_rewards)
    std_rewards = np.array(std_rewards)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Learning Curve Plot')
    ax.set_xlabel('Number of Episodes')
    ax.set_ylabel('Average Reward')
    ax.grid(True, which='both', alpha=.2)
    ax.plot(num_episodes, avg_rewards)
    ax.fill_between(num_episodes, avg_rewards-std_rewards, avg_rewards+std_rewards, alpha=.2)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(path + '/Learning_Curve_Plot.png')
    plt.close()


def make_Q_plot(num_episodes, avg_Qs, std_Qs, path):
    avg_Qs = np.array(avg_Qs)
    std_Qs = np.array(std_Qs)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Q Plot')
    ax.set_xlabel('Number of Episodes')
    ax.set_ylabel('Q')
    ax.grid(True, which='both', alpha=.2)
    ax.plot(num_episodes, avg_Qs)
    ax.fill_between(num_episodes, avg_Qs-std_Qs, avg_Qs+std_Qs, alpha=.2)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(path + '/Q_Plot.png')
    plt.close()