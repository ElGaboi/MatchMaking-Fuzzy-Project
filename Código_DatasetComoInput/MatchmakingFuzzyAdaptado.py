import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import os

# ============================================================
# FRONT-END
# ============================================================
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

# ============================================================
# VARIÁVEIS DE ENTRADA
# ============================================================
abates_por_rounds     = ctrl.Antecedent(np.arange(0, 1.2, 0.1),  'abates por rounds')
mortes_por_rounds     = ctrl.Antecedent(np.arange(0, 1.2, 0.1),  'mortes por rounds')
dano_por_round        = ctrl.Antecedent(np.arange(0, 120, 1),     'dano por round')
rounds_vivo_vitoria   = ctrl.Antecedent(np.arange(0, 100, 1),     'rounds ganho enquanto vivo (%)')
clutchs_ganhos        = ctrl.Antecedent(np.arange(0, 12, 1),      'clutchs ganhos')
classificacao         = ctrl.Antecedent(np.arange(0, 1.5, 0.1),   'classificacao')

# VARIÁVEL DE SAÍDA
grau_consistencia = ctrl.Consequent(np.arange(0, 10, 0.1), 'grau de consistencia')

# ============================================================
# FUNÇÕES DE PERTINÊNCIA
# ============================================================
abates_por_rounds['baixa']  = fuzz.trapmf(abates_por_rounds.universe, [-1, 0, 0.20, 0.50])
abates_por_rounds['media']  = fuzz.trimf(abates_por_rounds.universe,  [0.40, 0.60, 0.80])
abates_por_rounds['alta']   = fuzz.trapmf(abates_por_rounds.universe, [0.70, 0.90, 1.30, 1.40])

mortes_por_rounds['baixa']  = fuzz.trapmf(mortes_por_rounds.universe, [-1, 0, 0.4, 0.6])
mortes_por_rounds['media']  = fuzz.trimf(mortes_por_rounds.universe,  [0.50, 0.70, 0.90])
mortes_por_rounds['alta']   = fuzz.trapmf(mortes_por_rounds.universe, [0.80, 0.90, 1.20, 1.30])

dano_por_round['baixo']     = fuzz.trapmf(dano_por_round.universe, [-1, 0, 40, 60])
dano_por_round['medio']     = fuzz.trimf(dano_por_round.universe,  [50, 75, 100])
dano_por_round['alto']      = fuzz.trapmf(dano_por_round.universe, [90, 110, 120, 160])

rounds_vivo_vitoria['baixa'] = fuzz.trapmf(rounds_vivo_vitoria.universe, [-1, 0, 35, 45])
rounds_vivo_vitoria['media'] = fuzz.trimf(rounds_vivo_vitoria.universe,  [40, 50, 60])
rounds_vivo_vitoria['alta']  = fuzz.trapmf(rounds_vivo_vitoria.universe, [55, 65, 100, 110])

clutchs_ganhos['baixo']     = fuzz.trapmf(clutchs_ganhos.universe, [-1, 0, 1, 2])
clutchs_ganhos['medio']     = fuzz.trimf(clutchs_ganhos.universe,  [1, 3, 5])
clutchs_ganhos['alto']      = fuzz.trapmf(clutchs_ganhos.universe, [4, 6, 12, 13])

classificacao['baixo']      = fuzz.trapmf(classificacao.universe, [-1, 0, 0.50, 1])
classificacao['medio']      = fuzz.trimf(classificacao.universe,  [0.8, 1.0, 1.2])
classificacao['alto']       = fuzz.trapmf(classificacao.universe, [1.1, 1.3, 1.4, 1.6])

grau_consistencia['muito inconsistente'] = fuzz.trapmf(grau_consistencia.universe, [-1, 0, 1.5, 3])
grau_consistencia['inconsistente']       = fuzz.trimf(grau_consistencia.universe,  [2, 3.5, 5])
grau_consistencia['neutro']              = fuzz.trimf(grau_consistencia.universe,  [4, 5.5, 7])
grau_consistencia['consistente']         = fuzz.trimf(grau_consistencia.universe,  [6, 7.5, 9])
grau_consistencia['muito consistente']   = fuzz.trapmf(grau_consistencia.universe, [8, 9, 10, 11])

# ============================================================
# BASE DE REGRAS
# ============================================================
regras = [
    ctrl.Rule(abates_por_rounds['alta']  & mortes_por_rounds['baixa'] & dano_por_round['alto'],  grau_consistencia['muito consistente']),
    ctrl.Rule(abates_por_rounds['alta']  & mortes_por_rounds['baixa'] & dano_por_round['medio'], grau_consistencia['consistente']),
    ctrl.Rule(abates_por_rounds['alta']  & mortes_por_rounds['baixa'] & dano_por_round['baixo'], grau_consistencia['neutro']),
    ctrl.Rule(abates_por_rounds['alta']  & mortes_por_rounds['media'] & dano_por_round['alto'],  grau_consistencia['consistente']),
    ctrl.Rule(abates_por_rounds['alta']  & mortes_por_rounds['media'] & dano_por_round['medio'], grau_consistencia['consistente']),
    ctrl.Rule(abates_por_rounds['alta']  & mortes_por_rounds['media'] & dano_por_round['baixo'], grau_consistencia['inconsistente']),
    ctrl.Rule(abates_por_rounds['alta']  & mortes_por_rounds['alta']  & dano_por_round['alto'],  grau_consistencia['neutro']),
    ctrl.Rule(abates_por_rounds['alta']  & mortes_por_rounds['alta']  & dano_por_round['medio'], grau_consistencia['inconsistente']),
    ctrl.Rule(abates_por_rounds['alta']  & mortes_por_rounds['alta']  & dano_por_round['baixo'], grau_consistencia['inconsistente']),
    ctrl.Rule(abates_por_rounds['media'] & mortes_por_rounds['baixa'] & dano_por_round['alto'],  grau_consistencia['inconsistente']),
    ctrl.Rule(abates_por_rounds['media'] & mortes_por_rounds['baixa'] & dano_por_round['medio'], grau_consistencia['neutro']),
    ctrl.Rule(abates_por_rounds['media'] & mortes_por_rounds['baixa'] & dano_por_round['baixo'], grau_consistencia['inconsistente']),
    ctrl.Rule(abates_por_rounds['media'] & mortes_por_rounds['media'] & dano_por_round['alto'],  grau_consistencia['neutro']),
    ctrl.Rule(abates_por_rounds['media'] & mortes_por_rounds['media'] & dano_por_round['medio'], grau_consistencia['neutro']),
    ctrl.Rule(abates_por_rounds['media'] & mortes_por_rounds['media'] & dano_por_round['baixo'], grau_consistencia['inconsistente']),
    ctrl.Rule(abates_por_rounds['media'] & mortes_por_rounds['alta']  & dano_por_round['alto'],  grau_consistencia['inconsistente']),
    ctrl.Rule(abates_por_rounds['media'] & mortes_por_rounds['alta']  & dano_por_round['medio'], grau_consistencia['inconsistente']),
    ctrl.Rule(abates_por_rounds['media'] & mortes_por_rounds['alta']  & dano_por_round['baixo'], grau_consistencia['muito inconsistente']),
    ctrl.Rule(abates_por_rounds['baixa'] & mortes_por_rounds['baixa'] & dano_por_round['alto'],  grau_consistencia['neutro']),
    ctrl.Rule(abates_por_rounds['baixa'] & mortes_por_rounds['baixa'] & dano_por_round['medio'], grau_consistencia['inconsistente']),
    ctrl.Rule(abates_por_rounds['baixa'] & mortes_por_rounds['baixa'] & dano_por_round['baixo'], grau_consistencia['muito inconsistente']),
    ctrl.Rule(abates_por_rounds['baixa'] & mortes_por_rounds['media'] & dano_por_round['alto'],  grau_consistencia['inconsistente']),
    ctrl.Rule(abates_por_rounds['baixa'] & mortes_por_rounds['media'] & dano_por_round['medio'], grau_consistencia['muito inconsistente']),
    ctrl.Rule(abates_por_rounds['baixa'] & mortes_por_rounds['media'] & dano_por_round['baixo'], grau_consistencia['muito inconsistente']),
    ctrl.Rule(abates_por_rounds['baixa'] & mortes_por_rounds['alta']  & dano_por_round['alto'],  grau_consistencia['muito inconsistente']),
    ctrl.Rule(abates_por_rounds['baixa'] & mortes_por_rounds['alta']  & dano_por_round['medio'], grau_consistencia['muito inconsistente']),
    ctrl.Rule(abates_por_rounds['baixa'] & mortes_por_rounds['alta']  & dano_por_round['baixo'], grau_consistencia['muito inconsistente']),
    # Regras extras modificadoras
    ctrl.Rule(classificacao['alto']  & rounds_vivo_vitoria['alta'],                                              grau_consistencia['muito consistente']),
    ctrl.Rule(classificacao['alto']  & rounds_vivo_vitoria['media'],                                             grau_consistencia['consistente']),
    ctrl.Rule(classificacao['baixo'] & rounds_vivo_vitoria['baixa'],                                             grau_consistencia['muito inconsistente']),
    #ctrl.Rule(clutchs_ganhos['alto'] & (abates_por_rounds['media'] | dano_por_round['alto']),                    grau_consistencia['consistente']),
    #ctrl.Rule(clutchs_ganhos['baixo'] & (abates_por_rounds['baixa'] | dano_por_round['baixo']),                  grau_consistencia['inconsistente']),
    #ctrl.Rule(classificacao['medio'] & rounds_vivo_vitoria['alta'] & clutchs_ganhos['alto'],                     grau_consistencia['consistente']),
]

grau_consistencia_ctrl = ctrl.ControlSystem(regras)

# ============================================================
# LEITURA E ADAPTAÇÃO DO DATASET DE CS
# ============================================================

base_path = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_path, "Counter_Strike_all_Time_best_Players_Stats.csv")
df = pd.read_csv(csv_path)

# Corrige nome de coluna com espaço não-quebrado
df.columns = df.columns.str.replace('\xa0', ' ')

# --- Derivação das colunas necessárias ---

# kills e deaths a partir de K/D e K-D Diff
# Equações: kills/deaths = K/D  e  kills - deaths = K-D Diff
# => deaths = K-D Diff / (K/D - 1)  quando K/D != 1
mask = df['K/D'] != 1.0
df.loc[mask,  'deaths'] = df.loc[mask,  'K-D Diff'] / (df.loc[mask,  'K/D'] - 1)
df.loc[~mask, 'deaths'] = df.loc[~mask, 'Rounds'] * 0.7   # fallback para K/D == 1
df['kills'] = df['deaths'] * df['K/D']

df['abates_por_rounds']   = df['kills'] / df['Rounds']
df['mortes_por_rounds']   = df['deaths'] / df['Rounds']

# Rating1.0 já está no universo [0, 1.5] da variável 'classificacao'
df['classificacao']       = df['Rating1.0']

# Estimativa de dano por round a partir de abates e rating
# (proxy razoável dado que ADR e K/D são altamente correlacionados em CS)
df['dano_por_round']      = (df['Rating1.0'] * 75).clip(0, 119)

# Estimativa de rounds ganho enquanto vivo: sobrevivência ponderada pelo rating
rvv_raw = (1 - df['mortes_por_rounds']) * df['Rating1.0']
df['rounds_vivo_vitoria'] = (
    (rvv_raw - rvv_raw.min()) / (rvv_raw.max() - rvv_raw.min()) * 80 + 10
).clip(0, 99)


df['clutchs_ganhos']  = 3.0

# ============================================================
# COMPUTAÇÃO FUZZY POR JOGADOR
# ============================================================
resultados = []

for i, row in df.iterrows():
    simulador = ctrl.ControlSystemSimulation(grau_consistencia_ctrl)

    simulador.input['abates por rounds']              = row['abates_por_rounds']
    simulador.input['mortes por rounds']              = row['mortes_por_rounds']
    simulador.input['dano por round']                 = row['dano_por_round']
    simulador.input['rounds ganho enquanto vivo (%)'] = row['rounds_vivo_vitoria']
    simulador.input['classificacao']                  = row['classificacao']

    simulador.compute()

    if 'grau de consistencia' in simulador.output:
        resultados.append(simulador.output['grau de consistencia'])
    else:
        resultados.append(np.nan)

df['grau_consistencia'] = resultados
df.to_csv("MatchmakingFuzzyAdaptado_output.csv", index=False)

print(f"[OK] MatchmakingFuzzyAdaptado_output.csv gerado com {len(df)} jogadores.")
print(df[['Player', 'grau_consistencia']].head(10).to_string())

# ============================================================
# HISTOGRAMA DE SAÍDA
# ============================================================
plt.figure(figsize=(9, 5))

plt.hist(df['grau_consistencia'], bins=20, edgecolor='black', alpha=0.75)

plt.axvspan(0, 3,  alpha=0.15, label='Muito Inconsistente')
plt.axvspan(2, 5,  alpha=0.15, label='Inconsistente')
plt.axvspan(4, 7,  alpha=0.15, label='Neutro')
plt.axvspan(6, 9,  alpha=0.15, label='Consistente')
plt.axvspan(8, 10, alpha=0.15, label='Muito Consistente')

media = df['grau_consistencia'].mean()
plt.axvline(media, linestyle='--', linewidth=2, label=f"Média = {media:.2f}")

plt.xlabel("Grau de Consistência")
plt.ylabel("Número de Jogadores")
plt.title("Distribuição do Grau de Consistência dos Jogadores")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.35)
plt.tight_layout()
plt.show()