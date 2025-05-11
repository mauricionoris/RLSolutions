import matplotlib.pyplot as plt


def create_charts(epsilon_hist, q_table_history, rewards_per_episode):

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

    return figures

def show(figs):
    for fig in figs:
        fig.show()