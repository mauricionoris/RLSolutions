
## Introdução ao Ambiente CartPole e Discretização para Q-learning

### Ambiente CartPole

O ambiente **CartPole-v1**, disponível no OpenAI Gym, é um clássico problema de controle usado em aprendizado por reforço. O objetivo é manter um pêndulo invertido (uma haste conectada a um carrinho) equilibrado, aplicando forças para mover o carrinho para a esquerda ou direita.

* **Espaço de ações**: Discreto com 2 ações possíveis

  * `0`: mover para a esquerda
  * `1`: mover para a direita

* **Espaço de observações (estado)**: Contínuo, com 4 variáveis:

  * Posição do carrinho
  * Velocidade do carrinho
  * Ângulo da haste
  * Velocidade angular da haste

Essas variáveis formam um vetor contínuo de dimensão 4, tornando o espaço de estados infinito. Isso representa um desafio para algoritmos baseados em tabelas como o **Q-learning**, que requerem um número finito de estados.

---

### Discretização do Espaço de Estados

Para aplicar **Q-learning** no CartPole, é necessário **discretizar** o espaço de observações, mapeando os estados contínuos para estados discretos.

#### Etapas comuns da discretização:

1. **Definir os limites de cada dimensão**:
   Como os valores podem tender ao infinito em simulações longas, é comum definir limites razoáveis (com base em observações empíricas ou limites do ambiente).

2. **Dividir em bins (faixas)**:
   Cada dimensão é dividida em um número fixo de intervalos (bins). Exemplo:

   * Posição: 10 bins
   * Velocidade: 10 bins
   * Ângulo: 10 bins
   * Velocidade angular: 10 bins
     Resultando em $10^4 = 10.000$ estados discretos.

3. **Mapeamento para índice da Q-table**:
   Ao observar um estado contínuo, usamos os bins para converter esse vetor em um índice inteiro multidimensional que representa um estado discreto.

#### Exemplo em pseudocódigo:

```python
# Suponha que cada dimensão tenha sido dividida em 10 bins
discretized_state = tuple(
    np.digitize(obs[i], bins[i]) for i in range(len(obs))
)
q_value = q_table[discretized_state]
```

---

### Comparação com Ambientes Discretos (Taxi-v3)

O ambiente **Taxi-v3**, também do OpenAI Gym, é um exemplo de ambiente com:

* **Espaço de ações**: Discreto (6 ações: mover, pegar passageiro, deixar passageiro)
* **Espaço de estados**: Discreto (500 estados possíveis)

Neste caso, a Q-table é simplesmente uma matriz de tamanho `[n_states, n_actions] = [500, 6]`. Não há necessidade de discretização, pois cada observação já é um número inteiro representando um estado bem definido.

#### Vantagens do Taxi-v3:

* Facilidade de implementação de Q-learning puro
* Baixo custo computacional (pequena Q-table)
* Ideal para aprendizagem tabular e testes rápidos

#### Desafios do CartPole:

* Espaço de estados contínuo requer discretização
* Pode gerar Q-tables muito grandes
* Escolha do número de bins afeta desempenho e generalização

---

### Considerações Finais

| Característica            | CartPole                       | Taxi-v3                |
| ------------------------- | ------------------------------ | ---------------------- |
| Espaço de estados         | Contínuo (4D)                  | Discreto (500 estados) |
| Espaço de ações           | Discreto (2 ações)             | Discreto (6 ações)     |
| Necessita discretização   | Sim                            | Não                    |
| Uso em Q-learning tabular | Requer adaptação (discretizar) | Direto                 |
| Tamanho típico da Q-table | Pode ser muito grande          | Pequeno (500x6)        |

A discretização permite o uso de Q-learning em ambientes contínuos como o CartPole, mas é sensível ao número de bins usados. Já ambientes discretos como o Taxi-v3 são mais simples para algoritmos tabulares, sendo ideais para ensino e testes iniciais de agentes de RL.

