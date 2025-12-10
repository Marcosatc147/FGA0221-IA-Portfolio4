import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns

# Configurações
TAMANHO_GRID = 6 
BATERIA_INICIAL = 300 # Aumentei um pouco pela complexidade extra
# Códigos
LIMPO = 0
SUJEIRA = 1
OBSTACULO = -1 
ROBO = 2

class QuartoDesenhado:
    def __init__(self):
        self.tamanho = TAMANHO_GRID
        self.layout_fixo = np.array([
            [0,  0,  0, -1,  0,  0],
            [0, -1,  0, -1,  0, -1],
            [0, -1,  0,  0,  0,  0],
            [0,  0,  0, -1, -1,  0],
            [-1, 0, -1,  0,  0,  0],
            [0,  0,  0,  0, -1,  0]
        ])
        # Lista fixa de sujeiras para criar a máscara
        self.posicoes_sujeira_originais = [
            (0, 1), (0, 5), (2, 2), (2, 5),
            (4, 1), (4, 3), (5, 0), (5, 3)
        ]
        self.posicao_inicial = (0, 0)

    def get_sujeira_mask(self):
        # Cria uma tupla de 0s e 1s representando quais sujeiras ainda existem
        # Exemplo: (1, 1, 0, 1...) significa que a 3ª sujeira da lista foi limpa
        mask = []
        for x, y in self.posicoes_sujeira_originais:
            if self.grid[x, y] == SUJEIRA:
                mask.append(1)
            else:
                mask.append(0)
        return tuple(mask)

    def reset(self):
        self.grid = self.layout_fixo.copy()
        for (x, y) in self.posicoes_sujeira_originais:
            self.grid[x, y] = SUJEIRA
        self.posicao_robo = self.posicao_inicial
        self.bateria = BATERIA_INICIAL
        # O ESTADO AGORA É: (x, y, (1,1,1,1,1,1,1,1))
        return (self.posicao_robo[0], self.posicao_robo[1], self.get_sujeira_mask())

    def step(self, acao):
        x, y = self.posicao_robo
        if acao == 0: x = max(0, x-1)
        elif acao == 1: x = min(self.tamanho-1, x+1)
        elif acao == 2: y = max(0, y-1)
        elif acao == 3: y = min(self.tamanho-1, y+1)
        
        nova_pos = (x, y)
        recompensa = -1 
        self.bateria -= 1
        
        conteudo = self.grid[nova_pos]
        
        if conteudo == OBSTACULO:
            recompensa = -5
            nova_pos = self.posicao_robo 
        elif conteudo == SUJEIRA:
            recompensa = 50
            self.grid[nova_pos] = LIMPO
            
        self.posicao_robo = nova_pos
        
        sujeira_restante = np.any(self.grid == SUJEIRA)
        if not sujeira_restante:
            recompensa += 200 # Bônus maior para incentivar limpar TUDO
            
        done = self.bateria <= 0 or not sujeira_restante
        
        # Retorna o novo estado COMPLETO com a máscara atualizada
        return (nova_pos[0], nova_pos[1], self.get_sujeira_mask()), recompensa, done

# --- Treinamento ---
print("Treinando Robô (Estado com Memória de Sujeira)...")

# Q-Table agora é um dicionário para lidar com o estado complexo
# Chave: (x, y, mask_tuple) -> Valor: array de 4 ações
q_table = {}

def get_q_values(state):
    if state not in q_table:
        q_table[state] = np.zeros(4)
    return q_table[state]

alpha = 0.1
gamma = 0.95
epsilon = 1.0
decaimento_epsilon = 0.999 # Decaimento bem lento pois há mais estados para visitar

env = QuartoDesenhado()
historico = []

# Aumentei episódios pois o espaço de estados aumentou
for ep in range(15000): 
    estado = env.reset()
    done = False
    total = 0
    
    while not done:
        # Acessa valores Q usando o estado completo
        q_vals = get_q_values(estado)
        
        if random.uniform(0, 1) < epsilon:
            acao = random.randint(0, 3)
        else:
            acao = np.argmax(q_vals)
            
        novo_estado, reward, done = env.step(acao)
        
        antigo_val = q_vals[acao]
        futuro_val = np.max(get_q_values(novo_estado))
        
        # Atualiza o dicionário
        q_table[estado][acao] = antigo_val + alpha * (reward + gamma * futuro_val - antigo_val)
        
        estado = novo_estado
        total += reward
    
    if epsilon > 0.05: epsilon *= decaimento_epsilon
    historico.append(total)

    if ep % 1000 == 0:
        print(f"Episódio {ep}: Score = {total:.2f} (Epsilon: {epsilon:.2f})")

# --- Gráficos ---
plt.figure(figsize=(10,5))
media = np.convolve(historico, np.ones(200)/200, mode='valid')
plt.plot(media)
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Evolução do Aprendizado (Com Estado de Memória)")
plt.savefig('proj3_resultado_v2.png')
print("Salvo: proj3_resultado_v2.png")

# --- AUDITORIA: A VOLTA DA VITÓRIA ---
print("\n--- Iniciando Auditoria do Melhor Caminho ---")

# 1. Configurar ambiente para teste (sem aleatoriedade)
estado = env.reset()
done = False
path = [env.posicao_robo] # Guardar o caminho para desenhar
steps_log = []

while not done:
    # Pegar a MELHOR ação (Greedy)
    q_vals = get_q_values(estado)
    acao = np.argmax(q_vals)
    
    # Executar
    novo_estado, reward, done = env.step(acao)
    
    # Logar para verificação
    acoes_nome = ["Cima", "Baixo", "Esq", "Dir"]
    steps_log.append(f"Pos: {estado[0:2]} -> Ação: {acoes_nome[acao]} -> Reward: {reward}")
    
    estado = novo_estado
    path.append(env.posicao_robo)

# 2. Imprimir o Log passo a passo no terminal
print("\nPasso a Passo do Robô Treinado:")
for s in steps_log:
    print(s)

# 3. Desenhar a Rota Final
plt.figure(figsize=(6,6))

# Desenhar o Grid base
grid_visual = env.layout_fixo.copy()
# Marcar obstáculos com cor diferente
plt.imshow(grid_visual, cmap="Greys", origin="upper")

# Separar X e Y do caminho para plotar
y_coords = [p[0] for p in path] # Matplotlib usa Y invertido/trocado em matrizes as vezes, mas vamos testar direto
x_coords = [p[1] for p in path]

# Desenhar a linha do caminho
plt.plot(x_coords, y_coords, marker='o', color='blue', linewidth=2, label='Trajetória')
plt.scatter(x_coords[0], y_coords[0], color='green', s=200, label='Início', zorder=5)
plt.scatter(x_coords[-1], y_coords[-1], color='red', s=200, label='Fim', zorder=5)

# Marcar onde as sujeiras estavam
sujeiras_x = [p[1] for p in env.posicoes_sujeira_originais]
sujeiras_y = [p[0] for p in env.posicoes_sujeira_originais]
plt.scatter(sujeiras_x, sujeiras_y, marker='x', color='orange', s=100, label='Sujeiras', zorder=4)

plt.legend()
plt.title(f"Rota Ótima Aprendida (Score Final: {sum(historico[-10:])/10:.0f})")
plt.grid(True, alpha=0.3)

# Salvar
plt.savefig('proj3_rota_final.png')
print("\nAuditoria salva: proj3_rota_final.png")