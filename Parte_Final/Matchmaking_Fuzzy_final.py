import pandas as pd
import numpy as np
import random
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import itertools
import os



# MÉTODOS DE GERAÇÃO DE TIMES
def metodo_otimizado(df):
    organizado = df.sort_values(by="grau_consistencia").reset_index(drop=True)

    idx = random.randint(0, len(organizado) - 10)
    players_10 = organizado.iloc[idx:idx+10]
    ratings = players_10["grau_consistencia"].values

    best_score = float("inf")
    best_split = None

    for comb in itertools.combinations(range(10), 5):
        team_A = ratings[list(comb)]
        team_B = np.delete(ratings, comb)

        media_diff = abs(np.mean(team_A) - np.mean(team_B))
        variancia = max(np.std(team_A), np.std(team_B))

        score = media_diff * 0.7 + variancia * 0.3

        if score < best_score:
            best_score = score
            best_split = comb

    time_A = ratings[list(best_split)]
    time_B = np.delete(ratings, best_split)

    return time_A, time_B

def metodo_aleatorio(df):
    players = df.sample(n=10, replace=False).copy()

    ratings = np.array(players["grau_consistencia"], dtype=float)

    ratings = ratings.copy()
    np.random.shuffle(ratings)

    return ratings[:5], ratings[5:]


# SISTEMA FUZZY

def criar_sistema_fuzzy():
    diferenca = ctrl.Antecedent(np.arange(-2,2.01,0.05),'dif')
    variancia = ctrl.Antecedent(np.arange(0.00,2.01, 0.05), 'var')
    qualidade = ctrl.Consequent(np.arange(0,10.1,0.1),'qualidade')

    diferenca['ruim'] = fuzz.trapmf(diferenca.universe, [-2,-2,-1.2,-0.8])
    diferenca['leve'] = fuzz.trimf(diferenca.universe, [-1.2,0,1.2])
    diferenca['boa'] = fuzz.trapmf(diferenca.universe, [0.8,1.2,2,2])

    variancia['baixa'] = fuzz.trapmf(variancia.universe, [0,0,0.2,0.5])
    variancia['media'] = fuzz.trimf(variancia.universe, [0.3,0.8,1.3])
    variancia['alta'] = fuzz.trapmf(variancia.universe, [1,1.5,2,2])

    qualidade['ruim'] = fuzz.trimf(qualidade.universe, [0,2,4])
    qualidade['boa'] = fuzz.trimf(qualidade.universe, [3,5.5,8])
    qualidade['excelente'] = fuzz.trimf(qualidade.universe, [7,9,10])

    regras = [
        ctrl.Rule(diferenca['ruim'], qualidade['ruim']),
        ctrl.Rule(variancia['alta'], qualidade['ruim']),
        ctrl.Rule(diferenca['leve'] & variancia['baixa'], qualidade['excelente']),
        ctrl.Rule(diferenca['leve'] & variancia['media'], qualidade['boa']),
        ctrl.Rule(diferenca['boa'] & variancia['baixa'], qualidade['boa']),
    ]

    sistema = ctrl.ControlSystem(regras)
    return sistema


def avaliar_partida(time_A, time_B, sistema):
    sim = ctrl.ControlSystemSimulation(sistema)

    dif = np.mean(time_A) - np.mean(time_B)
    var = max(np.std(time_A), np.std(time_B))

    dif = np.clip(dif, -2.0, 2.0)
    var = np.clip(var,  0.0, 2.0)

    # Evita var=0 exato que causa falha de ativação nas regras
    if var == 0.0:
        var = 0.01

    sim.input['dif'] = dif
    sim.input['var'] = var

    try:
        sim.compute()
        return sim.output['qualidade']
    except KeyError:
        return 5.0

# FUNÇÃO DE SIMULAÇÃO (NOVO)


def simular_metodos(df, n=100):
    sistema = criar_sistema_fuzzy()

    resultados_otimizado = []
    resultados_aleatorio = []

    for _ in range(n):
        tA, tB = metodo_otimizado(df)
        resultados_otimizado.append(avaliar_partida(tA, tB, sistema))

        tA, tB = metodo_aleatorio(df)
        resultados_aleatorio.append(avaliar_partida(tA, tB, sistema))

    
    # GRÁFICO COMPARATIVO
    

    plt.figure(figsize=(10,5))

    plt.hist(resultados_otimizado, bins=20, alpha=0.6, label="Otimizado")
    plt.hist(resultados_aleatorio, bins=20, alpha=0.6, label="Aleatório")

    plt.xlabel("Qualidade do Pareamento")
    plt.ylabel("Frequência")
    plt.title("Comparação: Método Otimizado vs Aleatório")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()

    print(f"Média Otimizado: {np.mean(resultados_otimizado):.2f}")
    print(f"Média Aleatório: {np.mean(resultados_aleatorio):.2f}")


########################## MAIN #######################################
#Lê o documento CSV


base_path = os.path.dirname(__file__)
csv_path = os.path.join(base_path, "MatchmakingFuzzyAdaptado_output.csv")

df = pd.read_csv(csv_path)

METODO = 'otimizado'

# Seleciona e embaralha os 10 jogadores antes da divisão
players_10 = df.sample(n=10, replace=False).copy()
ratings = np.array(players_10["grau_consistencia"], dtype=float)
np.random.shuffle(ratings)


if METODO == "otimizado":
    best_diff = float("inf")
    best_split = None

    for comb in itertools.combinations(range(10), 5):
        team_A = ratings[list(comb)]
        team_B = np.delete(ratings, comb)

        media_diff = abs(np.mean(team_A) - np.mean(team_B))
        variancia = max(np.std(team_A), np.std(team_B))
        score = media_diff + variancia

        if score < best_diff:
            best_diff = score
            best_split = comb

    time_A = ratings[list(best_split)]
    time_B = np.delete(ratings, best_split)
else:
    # Divisão puramente aleatória — ratings já embaralhados no bloco acima
    time_A = ratings[:5]
    time_B = ratings[5:]

#Media dos times
media_A = np.mean(time_A)
media_B = np.mean(time_B)

#Desvio padrão interno de cada time
desvio_A = np.std(time_A, ddof=0)
desvio_B = np.std(time_B, ddof=0)

dif_medias = media_A - media_B
dispersao_partida = max(desvio_A, desvio_B)

metricas = {
    "media_A": media_A,
    "media_B": media_B,
    "diff_medias": dif_medias,
    "desvio_A": desvio_A,
    "desvio_B": desvio_B,
    "dispersao_partida": dispersao_partida
}

#Variaveis de entrada
diferenca_das_equipes = ctrl.Antecedent(np.arange(-2,2.01,0.05),'diferença das equipes')
variancia_interna = ctrl.Antecedent(np.arange(0.00,2.01, 0.05), 'variancia interna das equipes')

#Variaveis de Saída
qualidade_pareamento = ctrl.Consequent(np.arange(0,10.1,0.1),'qualidade do pareamento')

#Funções de pertinência
diferenca_das_equipes['B muito melhor'] = fuzz.trapmf(diferenca_das_equipes.universe, [-2,-2,-1.2,-0.8])
diferenca_das_equipes['B melhor'] = fuzz.trimf(diferenca_das_equipes.universe, [-1.2,-0.6,-0.1])
diferenca_das_equipes['equilibrado'] = fuzz.trimf(diferenca_das_equipes.universe, [-0.3,0,0.3])
diferenca_das_equipes['A melhor'] = fuzz.trimf(diferenca_das_equipes.universe, [0.1,0.6,1.2])
diferenca_das_equipes['A muito melhor'] = fuzz.trapmf(diferenca_das_equipes.universe, [0.8,1.2,2,2])

variancia_interna['muito baixa'] = fuzz.trapmf(variancia_interna.universe, [0,0,0.15,0.3])
variancia_interna['baixa'] = fuzz.trimf(variancia_interna.universe, [0.2,0.4,0.6])   
variancia_interna['moderada'] = fuzz.trimf(variancia_interna.universe, [0.5,0.8,1.1])
variancia_interna['alta'] = fuzz.trimf(variancia_interna.universe, [1,1.3,1.6])
variancia_interna['muito alta'] = fuzz.trapmf(variancia_interna.universe, [1.4,1.6,2,2])

qualidade_pareamento['muito ruim'] = fuzz.trapmf(qualidade_pareamento.universe, [0,0,1.5,3])
qualidade_pareamento['ruim'] = fuzz.trimf(qualidade_pareamento.universe, [2,3.5,5])
qualidade_pareamento['boa'] = fuzz.trimf(qualidade_pareamento.universe, [4.5,6.5,8])
qualidade_pareamento['muito boa'] = fuzz.trapmf(qualidade_pareamento.universe, [7,8.5,10,10])

#Base de regras
rule1 = ctrl.Rule(diferenca_das_equipes['A muito melhor'] | diferenca_das_equipes['B muito melhor'], qualidade_pareamento['muito ruim'])
rule2 = ctrl.Rule(diferenca_das_equipes['equilibrado'] & variancia_interna['muito alta'], qualidade_pareamento['ruim'])
rule3 = ctrl.Rule(diferenca_das_equipes['equilibrado'] & variancia_interna['muito baixa'], qualidade_pareamento['muito boa'])
rule4 = ctrl.Rule((diferenca_das_equipes['A melhor'] | diferenca_das_equipes['B melhor']) & variancia_interna['baixa'], qualidade_pareamento['boa'] )
rule5 = ctrl.Rule((diferenca_das_equipes['A melhor'] | diferenca_das_equipes['B melhor']) & variancia_interna['alta'], qualidade_pareamento['ruim'])
rule6 = ctrl.Rule(diferenca_das_equipes['equilibrado'] & variancia_interna['alta'], qualidade_pareamento['ruim'])
rule7 = ctrl.Rule(diferenca_das_equipes['equilibrado'] & variancia_interna['baixa'], qualidade_pareamento['muito boa'])
rule8 = ctrl.Rule(diferenca_das_equipes['equilibrado'] & variancia_interna['moderada'], qualidade_pareamento['muito boa'])
rule9 = ctrl.Rule((diferenca_das_equipes['A melhor'] | diferenca_das_equipes['B melhor']) & variancia_interna['moderada'], qualidade_pareamento['boa'])

#Gerando saída do Sistema Fuzzy
matchmaking_ctrl = ctrl.ControlSystem([rule1,rule2,rule3,rule4,rule5,rule6,rule7,rule8,rule9])
sim = ctrl.ControlSystemSimulation(matchmaking_ctrl)

sim.input['diferença das equipes'] = dif_medias
sim.input['variancia interna das equipes'] = dispersao_partida

sim.compute()

#qualidade = sim.output['qualidade do pareamento]
#print(f"Qualidade do pareamento Fuzzy: {qualidade: .2f}")

#Ativação das regras em variancia interna:
fig, ax = plt.subplots()
for term in variancia_interna.terms:
    ax.plot(
        variancia_interna.universe,
        variancia_interna[term].mf,
        linewidth=2,
        label=term   )
    
valorvar = dispersao_partida 

ax.axvline(
    valorvar,
    color="black",
    linestyle="--",
    linewidth=2,
    label=f"saída = {valorvar:.2f}"
)    

ax.set_title("")
ax.set_xlabel("Variancia")
ax.set_ylabel("Grau de pertinência")
ax.legend()
ax.grid(True)

plt.show()

#Ativação das regras em diferença das equipes
fig, ax = plt.subplots()
for term in diferenca_das_equipes.terms:
    ax.plot(
        diferenca_das_equipes.universe,
        diferenca_das_equipes[term].mf,
        linewidth=2,
        label=term   )
    
valordif = dif_medias 

ax.axvline(
    valordif,
    color="black",
    linestyle="--",
    linewidth=2,
    label=f"saída = {valordif:.2f}"
)    

ax.set_title("")
ax.set_xlabel("Diferença das equipes")
ax.set_ylabel("Grau de pertinência")
ax.legend()
ax.grid(True)

plt.show()

#PLOT GRÁFICO DA SAÍDA:
fig, ax = plt.subplots()
for term in qualidade_pareamento.terms:
    ax.plot(
        qualidade_pareamento.universe,
        qualidade_pareamento[term].mf,
        linewidth=2,
        label=term   )
    
valor = sim.output['qualidade do pareamento']    

ax.axvline(
    valor,
    color="black",
    linestyle="--",
    linewidth=2,
    label=f"saída = {valor:.2f}"
)    

ax.set_title("")
ax.set_xlabel("Qualidade do Pareamento")
ax.set_ylabel("Grau de pertinência")
ax.legend()
ax.grid(True)

plt.show()


#1) Gráfico de barras com desvio padrão

equipes = ["Time A", "Time B"]
medias = [media_A, media_B]
desvios = [desvio_A, desvio_B]

plt.figure()
plt.bar(equipes, medias, yerr=desvios, capsize=10)
plt.ylabel("Skill")
plt.title(f"Média e Desvio Padrão por Equipe")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()

    
#2) Boxplot das equipes

plt.figure()
plt.boxplot([time_A,time_B], labels=["Time A", "Time B"], showmeans=True)
plt.ylabel("Skill")
plt.title(f"Distribuição interna das equipes")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()


#3) Histograma geral 

todos_players = np.concatenate([time_A,time_B])

plt.figure()
plt.hist(todos_players, bins=10)
plt.xlabel("Habilidade")
plt.ylabel("Frequência")
plt.title(f"Distribuição geral dos jogadores")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()


#print(f"Média Time A: {media_A:.2f}")
#print(f"Média Time B: {media_B:.2f}")
#print(f"Desvio Padrão Time A: {desvio_A:.2f}")
#print(f"Desvio Padrão Time B: {desvio_B:.2f}")
simular_metodos(df, n=200)



