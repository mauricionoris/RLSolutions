import matplotlib.pyplot as plt
import numpy as np


def policy_view(grid,policy):
        policy_matrix = np.zeros(grid.size, dtype=object)

        # Símbolos para as direções
        s = {   0: "↑",  1: "→",   2: "↓",   3: "←"  }

        h, w = grid.size

        policy_matrix[grid.start] =  'S'

        policy_matrix[grid.goal] =  'G'

        for o in grid.obstacles:
            policy_matrix[o] = 'X'

        for p in policy.keys():
            policy_matrix[p] = s[policy[p]]

        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        # Criando a grade
        for i in range(h + 1):
            plt.axhline(i - 0.5, color='black', linestyle='-', linewidth=1)
        for j in range(w + 1):
            plt.axvline(j - 0.5, color='black', linestyle='-', linewidth=1)

        # Preenchendo com as ações
        for i in range(h):
            for j in range(w):
                ax.text(j, i, policy_matrix[i, j], ha='center', va='center', fontsize=20)

        plt.title('Política Ótima')
        plt.xlim(-0.5, w - 0.5)
        plt.ylim(h - 0.5, -0.5)
        plt.xticks(range(w))
        plt.yticks(range(h))
        plt.grid(False)
        plt.tight_layout()
        plt.show()



class Grid:
    def __init__(self, size, start, goal, obstacles):
        self.size = size
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        

policy =  {(0,1):2, (0,2):2, (1,1):2, (1,2):3}
mygrid = Grid((4,4),(0,0),(3,3),[(2,2),(3,2)])

policy_view(mygrid, policy)
