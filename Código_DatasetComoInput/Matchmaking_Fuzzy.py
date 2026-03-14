import pandas as pd
import numpy as np
import random
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


#Front-End
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams.update({
    "figure.figsize": (8, 4),
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

#Variáveis de entrada
abates_por_rounds = ctrl.Antecedent(np.arange(0, 1.2, 0.1), 'abates por rounds')
mortes_por_rounds = ctrl.Antecedent(np.arange(0,1.2, 0.1), 'mortes por rounds')
dano_por_round = ctrl.Antecedent(np.arange(0,120,1), 'dano por round')
rounds_vivo_vitoria = ctrl.Antecedent(np.arange(0,100,1), 'rounds ganho enquanto vivo (%)')
clutchs_ganhos = ctrl.Antecedent(np.arange(0,12, 1), 'clutchs ganhos')
classificacao = ctrl.Antecedent(np.arange(0, 1.5, 0.1), 'classificacao')

#variáveis de saída
grau_consistencia = ctrl.Consequent(np.arange(0, 10, 0.1), 'grau de consistencia')

############################################################
#Funções de pertinência para taxa de abates por rounds
abates_por_rounds['baixa'] = fuzz.trapmf(abates_por_rounds.universe, [-1, 0, 0.20, 0.50])
abates_por_rounds['media'] = fuzz.trimf(abates_por_rounds.universe, [0.40, 0.60, 0.80])
abates_por_rounds['alta'] = fuzz.trapmf(abates_por_rounds.universe, [0.70, 0.90, 1.30,1.40])

#Funções de pertinência para taxa de mortes por rounds
mortes_por_rounds['baixa'] = fuzz.trapmf(mortes_por_rounds.universe, [-1,0,0.4,0.6])
mortes_por_rounds['media'] = fuzz.trimf(mortes_por_rounds.universe, [0.50,0.70,0.90])
mortes_por_rounds['alta'] = fuzz.trapmf(mortes_por_rounds.universe,  [0.80,0.90,1.20,1.30])

#Funções de pertinencia para media de dano por rounds
dano_por_round['baixo'] = fuzz.trapmf(dano_por_round.universe, [-1,0,40,60])
dano_por_round['medio'] = fuzz.trimf(dano_por_round.universe, [50,75,100])
dano_por_round['alto'] = fuzz.trapmf(dano_por_round.universe, [90,110,120,160])

#Funções de pertinencia para porcentagem de rounds ganhos enquanto vivo
rounds_vivo_vitoria['baixa'] = fuzz.trapmf(rounds_vivo_vitoria.universe, [-1,0,35,45])
rounds_vivo_vitoria['media'] = fuzz.trimf(rounds_vivo_vitoria.universe, [40,50,60])
rounds_vivo_vitoria['alta'] = fuzz.trapmf(rounds_vivo_vitoria.universe, [55,65,100,110])

#Funções de pertinencia para clutchs ganhos
clutchs_ganhos['baixo'] = fuzz.trapmf(clutchs_ganhos.universe, [-1,0,1,2])
clutchs_ganhos['medio'] = fuzz.trimf(clutchs_ganhos.universe, [1,3,5])
clutchs_ganhos['alto'] = fuzz.trapmf(clutchs_ganhos.universe, [4,6,12,13])

#Funções de pertinencia para classificação/rating
classificacao['baixo'] = fuzz.trapmf(classificacao.universe,[-1,0,0.50,1])
classificacao['medio'] = fuzz.trimf(classificacao.universe,[0.8, 1.0,1.2])
classificacao['alto'] = fuzz.trapmf(classificacao.universe, [1.1,1.3,1.4,1.6])

#Funções de pertinência para consistencia
grau_consistencia['muito inconsistente'] = fuzz.trapmf(grau_consistencia.universe, [-1,0,1.5,3])
grau_consistencia['inconsistente'] = fuzz.trimf(grau_consistencia.universe, [2,3.5,5])
grau_consistencia['neutro'] = fuzz.trimf(grau_consistencia.universe, [4,5.5,7])
grau_consistencia['consistente'] = fuzz.trimf(grau_consistencia.universe, [6,7.5,9])
grau_consistencia['muito consistente'] = fuzz.trapmf(grau_consistencia.universe, [8,9,10,11])

#Base de regras
regra1 = ctrl.Rule(abates_por_rounds['alta'] & mortes_por_rounds['baixa'] & dano_por_round['alto'], grau_consistencia['muito consistente'])
regra2 = ctrl.Rule(abates_por_rounds['alta'] & mortes_por_rounds['baixa'] & dano_por_round['medio'], grau_consistencia['consistente'])
regra3 = ctrl.Rule(abates_por_rounds['alta'] & mortes_por_rounds['baixa'] & dano_por_round['baixo'], grau_consistencia['neutro'])
regra4 = ctrl.Rule(abates_por_rounds['alta'] & mortes_por_rounds['media'] & dano_por_round['alto'], grau_consistencia['consistente'])
regra5 = ctrl.Rule(abates_por_rounds['alta'] & mortes_por_rounds['media'] & dano_por_round['medio'], grau_consistencia['consistente'])
regra6 = ctrl.Rule(abates_por_rounds['alta'] & mortes_por_rounds['media'] & dano_por_round['baixo'], grau_consistencia['inconsistente'])
regra7 = ctrl.Rule(abates_por_rounds['alta'] & mortes_por_rounds['alta'] & dano_por_round['alto'], grau_consistencia['neutro'])
regra8 = ctrl.Rule(abates_por_rounds['alta'] & mortes_por_rounds['alta'] & dano_por_round['medio'], grau_consistencia['inconsistente'])
regra9 = ctrl.Rule(abates_por_rounds['alta'] & mortes_por_rounds['alta'] & dano_por_round['baixo'], grau_consistencia['inconsistente'])
regra10 = ctrl.Rule(abates_por_rounds['media'] & mortes_por_rounds['baixa'] & dano_por_round['alto'], grau_consistencia['inconsistente'])
regra11 = ctrl.Rule(abates_por_rounds['media'] & mortes_por_rounds['baixa'] & dano_por_round['medio'], grau_consistencia['neutro'])
regra12 = ctrl.Rule(abates_por_rounds['media'] & mortes_por_rounds['baixa'] & dano_por_round['baixo'], grau_consistencia['inconsistente'])
regra13 = ctrl.Rule(abates_por_rounds['media'] & mortes_por_rounds['media'] & dano_por_round['alto'], grau_consistencia['neutro'])
regra14 = ctrl.Rule(abates_por_rounds['media'] & mortes_por_rounds['media'] & dano_por_round['medio'], grau_consistencia['neutro'])
regra15 = ctrl.Rule(abates_por_rounds['media'] & mortes_por_rounds['media'] & dano_por_round['baixo'], grau_consistencia['inconsistente'])
regra16 = ctrl.Rule(abates_por_rounds['media'] & mortes_por_rounds['alta'] & dano_por_round['alto'], grau_consistencia['inconsistente'])
regra17 = ctrl.Rule(abates_por_rounds['media'] & mortes_por_rounds['alta'] & dano_por_round['medio'], grau_consistencia['inconsistente'])
regra18 = ctrl.Rule(abates_por_rounds['media'] & mortes_por_rounds['alta'] & dano_por_round['baixo'], grau_consistencia['muito inconsistente'])
regra19 = ctrl.Rule(abates_por_rounds['baixa'] & mortes_por_rounds['baixa'] & dano_por_round['alto'], grau_consistencia['neutro'])
regra20 = ctrl.Rule(abates_por_rounds['baixa'] & mortes_por_rounds['baixa'] & dano_por_round['medio'], grau_consistencia['inconsistente'])
regra21 = ctrl.Rule(abates_por_rounds['baixa'] & mortes_por_rounds['baixa'] & dano_por_round['baixo'], grau_consistencia['muito inconsistente'])
regra22 = ctrl.Rule(abates_por_rounds['baixa'] & mortes_por_rounds['media'] & dano_por_round['alto'], grau_consistencia['inconsistente'])
regra23 = ctrl.Rule(abates_por_rounds['baixa'] & mortes_por_rounds['media'] & dano_por_round['medio'], grau_consistencia['muito inconsistente'])
regra24 = ctrl.Rule(abates_por_rounds['baixa'] & mortes_por_rounds['media'] & dano_por_round['baixo'], grau_consistencia['muito inconsistente'])
regra25 = ctrl.Rule(abates_por_rounds['baixa'] & mortes_por_rounds['alta'] & dano_por_round['alto'], grau_consistencia['muito inconsistente'])
regra26 = ctrl.Rule(abates_por_rounds['baixa'] & mortes_por_rounds['alta'] & dano_por_round['medio'], grau_consistencia['muito inconsistente'])
regra27 = ctrl.Rule(abates_por_rounds['baixa'] & mortes_por_rounds['alta'] & dano_por_round['baixo'], grau_consistencia['muito inconsistente'])
#Regras extras modificadoras
regra28 = ctrl.Rule(classificacao['alto'] & rounds_vivo_vitoria['alta'], grau_consistencia['muito consistente'])
regra29 = ctrl.Rule(classificacao['alto'] & rounds_vivo_vitoria['media'], grau_consistencia['consistente'])
regra30 = ctrl.Rule(classificacao['baixo'] & rounds_vivo_vitoria['baixa'], grau_consistencia['muito inconsistente'])
regra31 = ctrl.Rule(clutchs_ganhos['alto'] & (abates_por_rounds['media'] | dano_por_round['alto']), grau_consistencia['consistente'])
regra32 = ctrl.Rule(clutchs_ganhos['baixo'] & (abates_por_rounds['baixa'] | dano_por_round['baixo']), grau_consistencia['inconsistente'])
regra33 = ctrl.Rule(classificacao['medio'] & rounds_vivo_vitoria['alta'] & clutchs_ganhos['alto'], grau_consistencia['consistente'])

#Sistema Fuzzy
grau_consistencia_ctrl = ctrl.ControlSystem([regra1,regra2,regra3,regra4,regra5,regra6,regra7,regra8,regra9,regra10,regra11,regra12,regra13,regra14,regra15,regra16,regra17,regra18,regra19,regra20,regra21,regra22,regra23,regra24,regra25,regra26,regra27,regra28,regra29,regra30,regra31,regra32,regra33])


#Input
#Leitura do Dataset
df = pd.read_csv("players_stats.csv")
resultados = []

for i, row in df.iterrows():
    simulador = ctrl.ControlSystemSimulation(grau_consistencia_ctrl)

    simulador.input['abates por rounds'] = row['abates_por_rounds']
    simulador.input['mortes por rounds'] = row['mortes_por_rounds']
    simulador.input['dano por round'] = row['dano_por_round']
    simulador.input['rounds ganho enquanto vivo (%)'] = row['rounds_vivo_vitoria']
    simulador.input['clutchs ganhos'] = row['clutchs_ganhos']
    simulador.input['classificacao'] = row['classificacao']

    simulador.compute()

    if 'grau de consistencia' in simulador.output:
        resultados.append(simulador.output['grau de consistencia'])
    else:
        resultados.append(np.nan)

df['grau_consistencia'] = resultados
df.to_csv("matchmaking_fuzzy_output.csv", index=False)

#PLOT histograma das saídas

plt.figure(figsize=(9,5))

plt.hist(
    df['grau_consistencia'],
    bins=20,
    edgecolor='black',
    alpha=0.75
)
#Regiões Fuzzy
plt.axvspan(0, 3, alpha=0.15, label='Muito Inconsistente')
plt.axvspan(2, 5, alpha=0.15, label='Inconsistente')
plt.axvspan(4, 7, alpha=0.15, label='Neutro')
plt.axvspan(6, 9, alpha=0.15, label='Consistente')
plt.axvspan(8, 10, alpha=0.15, label='Muito Consistente')

#Media
media = df['grau_consistencia'].mean()
plt.axvline(
    media,
    linestyle='--',
    linewidth=2,
    label=f"Média = {media:.2f}"
)


plt.xlabel("Grau de Consistência")
plt.ylabel("Número de Jogadores")
plt.title("Distribuição do Grau de Consistência dos Jogadores")

plt.grid(axis='y', linestyle='--', alpha=0.35)
plt.tight_layout()
plt.show()

input()



