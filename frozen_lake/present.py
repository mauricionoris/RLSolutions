import matplotlib.pyplot as plt
import numpy as np

def create_charts(epsilon_hist, q_table_history, rewards_per_episode, grid, policy, experiment_name):

    figures = []

    # 1. Decaimento do epsilon
    fig1, ax1 = plt.subplots()
    ax1.plot(epsilon_hist, color="red")
    ax1.set_title("Decaimento do epsilon")
    ax1.set_xlabel("Episódios")
    ax1.set_ylabel("epsilon")
    ax1.grid(True)
    figures.append(fig1)

    # 2. Evolução da média geral da Q-Table
    fig2, ax2 = plt.subplots()
    xi = range(len(q_table_history))
    ax2.plot(xi, q_table_history, color="blue")
    ax2.set_title("Evolução da Média Geral da Q-Table")
    ax2.set_xlabel("Episódios")
    ax2.set_ylabel("Média dos Valores Q")
    ax2.grid(True)
    figures.append(fig2)

    # 3. Total rewards por episódio
    fig3, ax3 = plt.subplots()
    ax3.plot(rewards_per_episode)
    ax3.set_xlabel("Episódios")
    ax3.set_ylabel("Total rewards")
    ax3.set_title("Total rewards per episode")
    figures.append(fig3)

    # 4. Média móvel dos rewards
    def media_movel(dados, janela):
        return [sum(dados[i:i+janela])/janela for i in range(len(dados) - janela + 1)]

    media_movel_ = media_movel(rewards_per_episode, janela=100)
    fig4, ax4 = plt.subplots()
    ax4.plot(media_movel_, color="black", alpha=0.9)
    ax4.set_title("Valor médio dos rewards")
    ax4.set_xlabel("Episódios")
    ax4.set_ylabel("média")
    ax4.grid(True)
    figures.append(fig4)

    # 3. Total rewards por episódio
    fig5, ax5 = plt.subplots()
    ax5 = policy_view(ax5, grid, policy)
    ax4.set_title('Política Ótima')
    ax4.set_xlabel("Episódios")
    ax4.set_ylabel("média")
    figures.append(fig5)


    return figures


def save_pic(figs, path, exp_name):
    p = [f'epsilon_decay.png',f'q_table_average.png' ,f'rewards_per_episode.png', f'moving_average_rewards.png',f'optimal_policy.png']

    for fig, filename in zip(figs, p):
        fig.savefig(f'{path}{exp_name}_{filename}')

def show(figs):
    for fig in figs:
        fig.show()

def close(figs):
    for fig in figs:
         plt.close(fig)

def policy_view(ax5 ,grid, policy):
        policy_matrix = np.zeros(grid.size, dtype=object)

        # Símbolos para as direções
        s = {   3: "↑",  2: "→",   1: "↓",   0: "←"  }
#       ['left', 'down', 'right', 'up'],

        data1 = np.argmax(policy, axis=1).reshape(-1, 1)
        reshaped = data1.reshape(grid.size)
        dict_4x12 = {(i, j): reshaped[i, j] for i in range(grid.size[0]) for j in range(grid.size[1])}
        for p in dict_4x12.keys():
            policy_matrix[p] = s[dict_4x12[p]]

        h, w = grid.size

        policy_matrix[grid.start] =  'S'

        policy_matrix[grid.goal] =  'G'


        
        for o in grid.obstacles:
            policy_matrix[o] = 'X'


     
 

        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        # Criando a grade
        for i in range(h + 1):
            ax5.axhline(i - 0.5, color='black', linestyle='-', linewidth=1)
        for j in range(w + 1):
            ax5.axvline(j - 0.5, color='black', linestyle='-', linewidth=1)

        # Preenchendo com as ações
        for i in range(h):
            for j in range(w):
                ax5.text(j, i, policy_matrix[i, j], ha='center', va='center', fontsize=20)

        ax5.set_xlim(-0.5, w - 0.5)
        ax5.set_ylim(h - 0.5, -0.5)
        ax5.set_xticks(range(w))
        ax5.set_yticks(range(h))
        ax5.grid(False)
        ax5._in_layout = True
        return ax5


