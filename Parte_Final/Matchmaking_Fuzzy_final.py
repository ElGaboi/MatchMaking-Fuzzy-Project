import pandas as pd
import numpy as np
import random
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


#Lê o documento CSV
df = pd.read_csv("matchmaking_fuzzy_output.csv")

#Extrai 10 jogadores do dataset 
players_10 = df.sample(n=10, replace=False, random_state=None)
ratings = players_10["grau_consistencia"].values
ratings.shape

#Separa em dois times
indices = np.arange(10)
np.random.shuffle(indices)

time_A_idx = indices[:5]
time_B_idx = indices[5:]

time_A = ratings[time_A_idx]
time_B = ratings[time_B_idx]

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
qualidade_partida = ctrl.Consequent(np.arange(0,10.1,0.1),'qualidade da partida')

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

qualidade_partida['muito ruim'] = fuzz.trapmf(qualidade_partida.universe, [0,0,1.5,3])
qualidade_partida['ruim'] = fuzz.trimf(qualidade_partida.universe, [2,3.5,5])
qualidade_partida['boa'] = fuzz.trimf(qualidade_partida.universe, [4.5,6.5,8])
qualidade_partida['muito boa'] = fuzz.trapmf(qualidade_partida.universe, [7,8.5,10,10])

#Base de regras
rule1 = ctrl.Rule(diferenca_das_equipes['A muito melhor'] | diferenca_das_equipes['B muito melhor'], qualidade_partida['muito ruim'])
rule2 = ctrl.Rule(diferenca_das_equipes['equilibrado'] & variancia_interna['muito alta'], qualidade_partida['ruim'])
rule3 = ctrl.Rule(diferenca_das_equipes['equilibrado'] & variancia_interna['muito baixa'], qualidade_partida['muito boa'])
rule4 = ctrl.Rule((diferenca_das_equipes['A melhor'] | diferenca_das_equipes['B melhor']) & variancia_interna['baixa'], qualidade_partida['boa'] )
rule5 = ctrl.Rule((diferenca_das_equipes['A melhor'] | diferenca_das_equipes['B melhor']) & variancia_interna['alta'], qualidade_partida['ruim'])
rule6 = ctrl.Rule(diferenca_das_equipes['equilibrado'] & variancia_interna['alta'], qualidade_partida['ruim'])
rule7 = ctrl.Rule(diferenca_das_equipes['equilibrado'] & variancia_interna['baixa'], qualidade_partida['muito boa'])
rule8 = ctrl.Rule(diferenca_das_equipes['equilibrado'] & variancia_interna['moderada'], qualidade_partida['muito boa'])
rule9 = ctrl.Rule((diferenca_das_equipes['A melhor'] | diferenca_das_equipes['B melhor']) & variancia_interna['moderada'], qualidade_partida['boa'])

#Gerando saída do Sistema Fuzzy
matchmaking_ctrl = ctrl.ControlSystem([rule1,rule2,rule3,rule4,rule5,rule6,rule7,rule8,rule9])
sim = ctrl.ControlSystemSimulation(matchmaking_ctrl)

sim.input['diferença das equipes'] = dif_medias
sim.input['variancia interna das equipes'] = dispersao_partida

sim.compute()

#qualidade = sim.output['qualidade da partida']
#print(f"Qualidade da partida Fuzzy: {qualidade: .2f}")

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
for term in qualidade_partida.terms:
    ax.plot(
        qualidade_partida.universe,
        qualidade_partida[term].mf,
        linewidth=2,
        label=term   )
    
valor = sim.output['qualidade da partida']    

ax.axvline(
    valor,
    color="black",
    linestyle="--",
    linewidth=2,
    label=f"saída = {valor:.2f}"
)    

ax.set_title("")
ax.set_xlabel("Qualidade da partida")
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





