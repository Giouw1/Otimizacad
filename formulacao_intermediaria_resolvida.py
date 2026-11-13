import gurobipy as gp
from gurobipy import GRB


def resolver_modelo_anterior(nome_cenario, dados):
  """Implementa a formulação anterior (baseada em Tr_i, Fad_i e Ex_{i,j} sem ordem k)."""
  dias = dados["dias"]
  disp = dados["disp"]
  exercicios = list(dados["alpha"].keys())
  alpha = dados["alpha"]
  a = dados["a"]
  b = dados["b"]
  # Limite máximo que um único exercício pode ocupar no dia (1.0 = sem limite, fiel à foto)
  teto_ex = dados.get("teto_por_exercicio", 1.0)

  model = gp.Model(nome_cenario)
  model.Params.OutputFlag = 0

  # ==========================================
  # 1. VARIÁVEIS DE DECISÃO
  # ==========================================
  # Ex_{i,j} >= 0 (Fração/intensidade do treino dedicada ao exercício j no dia i)
  Ex = model.addVars(
      dias, exercicios, lb=0.0, ub=teto_ex, vtype=GRB.CONTINUOUS, name="Ex"
  )
  # Tr_i <= 1 (Volume/Capacidade de treino realizada no dia i)
  Tr = model.addVars(dias, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="Tr")
  # Fad_i <= 1 (Fadiga acumulada no dia i)
  Fad = model.addVars(dias, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="Fad")

  # ==========================================
  # 2. FUNÇÃO OBJETIVO: max sum_i sum_j alpha_j * Ex_{i,j}
  # ==========================================
  model.setObjective(
      gp.quicksum(alpha[j] * Ex[i, j] for i in dias for j in exercicios),
      GRB.MAXIMIZE,
  )

  # ==========================================
  # 3. RESTRIÇÕES (Exatamente como na imagem)
  # ==========================================
  # Condição inicial da fadiga no Dia 1
  model.addConstr(Fad[dias[0]] == 0.0, name="Fad_Inicial")

  for i in dias:
    # Restrição da variável de treino (com disponibilidade do dia: Tr_i <= Disp_i <= 1)
    model.addConstr(Tr[i] <= disp[i], name=f"Max_Tr_d{i}")

    # Restrição da fadiga: Fad_i <= 1 (já garantido pelo ub=1.0, mas explícito aqui)
    model.addConstr(Fad[i] <= 1.0, name=f"Max_Fad_d{i}")

    # Restrição do treino levando em conta a fadiga: Tr_i <= 1 - Fad_i
    model.addConstr(Tr[i] <= 1.0 - Fad[i], name=f"Tr_Fad_d{i}")

    # Restrição dos exercícios (ignorando a ordem): sum_j Ex_{i,j} == Tr_i
    model.addConstr(
        gp.quicksum(Ex[i, j] for j in exercicios) == Tr[i],
        name=f"Soma_Ex_Tr_d{i}",
    )

  # Restrição da fadiga com base no treino: Fad_{i+1} = a * Tr_i + b * Fad_i
  for idx in range(len(dias) - 1):
    i_atual = dias[idx]
    i_prox = dias[idx + 1]
    model.addConstr(
        Fad[i_prox] == a * Tr[i_atual] + b * Fad[i_atual],
        name=f"Evol_Fad_d{i_prox}",
    )

  # ==========================================
  # 4. RESOLUÇÃO E IMPRESSÃO DOS RESULTADOS
  # ==========================================
  model.optimize()

  print("=" * 75)
  print(f"CENÁRIO: {nome_cenario}")
  if model.status == GRB.OPTIMAL:
    print(f"STATUS: Ótimo Encontrado | Valor FObj: {model.objVal:.4f}")
    print("=" * 75)
    for i in dias:
      status = "DISPONÍVEL" if disp[i] == 1 else "INDISPONÍVEL"
      print(
          f"Dia {i} ({status:<12}) | Fadiga: {Fad[i].X:.2f} | "
          f"Teto (1-Fad): {1 - Fad[i].X:.2f} | Treino (Tr_i): {Tr[i].X:.2f}"
      )
      for j in exercicios:
        if Ex[i, j].X > 1e-4:
          print(
              f"   -> Exercício: {j:<22} | Ex_{{i,j}} = {Ex[i, j].X:.2f} "
              f"(alpha = {alpha[j]})"
          )
    print("\n")
  else:
    print(f"Não foi possível encontrar solução ótima. Status: {model.status}\n")


# =============================================================================
# MOCK 1: SIMPLES (2 Exercícios, 3 Dias, Formulação 100% Pura da Imagem)
# =============================================================================
mock_simples = {
    "dias": [1, 2, 3],
    "disp": {1: 1, 2: 1, 3: 1},
    "a": 0.60,  # Cada unidade de treino gera 0.60 de fadiga para o dia seguinte
    "b": 0.30,  # 30% da fadiga anterior permanece no dia seguinte
    "teto_por_exercicio": (
        1.0
    ),  # Sem limite individual (mostra o efeito guloso no maior alpha)
    "alpha": {"Agachamento (Pernas)": 12.0, "Supino Reto (Peito)": 8.0},
}

# =============================================================================
# MOCK 2: COMPLEXO (7 Dias, 8 Exercícios, com limite de 0.35 por exercício no dia
# para forçar o modelo a escolher múltiplos exercícios para fechar a soma == Tr_i)
# =============================================================================
mock_complexo = {
    "dias": [1, 2, 3, 4, 5, 6, 7],
    "disp": {
        1: 1,
        2: 1,
        3: 0,
        4: 1,
        5: 1,
        6: 1,
        7: 0,
    },  # Descanso na Quarta (3) e Domingo (7)
    "a": 0.55,
    "b": 0.40,
    "teto_por_exercicio": (
        0.35
    ),  # Cada exercício pode preencher no máximo 35% do treino máximo
    "alpha": {
        "Agachamento_Livre": 15.0,
        "Terra_Romeno": 14.0,
        "Supino_Reto": 12.0,
        "Remada_Curvada": 12.0,
        "Desenvolvimento": 9.0,
        "Puxada_Alta": 9.0,
        "Cadeira_Extensora": 7.0,
        "Rosca_Direta": 6.0,
    },
}

if __name__ == "__main__":
  resolver_modelo_anterior("MOCK 1 - SIMPLES (Puro da Imagem)", mock_simples)
  resolver_modelo_anterior("MOCK 2 - COMPLEXO (Semana de 7 Dias)", mock_complexo)