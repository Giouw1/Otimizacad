import gurobipy as gp
from gurobipy import GRB


def resolver_modelo_treino(nome_cenario, dados):
  """Constrói e resolve o MILP de planejamento de treinos no Gurobi

  a partir de um dicionário de dados (Mock).
  """
  dias = dados["dias"]
  slots = dados["slots"]
  disp = dados["disp"]
  musculos = dados["musculos"]
  pref = dados["pref"]
  catalogo = dados["catalogo"]
  a_fad = dados["a_fad"]
  b_fad = dados["b_fad"]
  w_slot = dados["w_slot"]
  I_min = dados.get("I_min", 0.20)

  exercicios = list(catalogo.keys())

  # =========================================================================
  # PRÉ-PROCESSAMENTO: PRs -> 1RM, Fator I_hat e Carga Máxima W_max (Epley)
  # =========================================================================
  um_rm = {}
  I_hat = {}
  W_max = {}
  utilidade_base = {}

  for j, info in catalogo.items():
    # 1RM = PR_peso * (1 + PR_reps / 30)
    um_rm[j] = info["pr_peso"] * (1.0 + info["pr_reps"] / 30.0)
    # Fração de 1RM para as repetições fixadas no treino: 1 / (1 + reps_treino / 30)
    I_hat[j] = 1.0 / (1.0 + info["reps_treino"] / 30.0)
    # Carga 100% (em kg) para o número de repetições do treino
    W_max[j] = um_rm[j] * I_hat[j]
    # Recompensa muscular ponderada pelo fator de intensidade de 1RM (I_hat)
    utilidade_base[j] = I_hat[j] * sum(
        pref[m] * info["alpha"].get(m, 0.0) for m in musculos
    )

  # =========================================================================
  # INICIALIZAÇÃO E VARIÁVEIS DO GUROBI
  # =========================================================================
  model = gp.Model(nome_cenario)
  model.Params.OutputFlag = 0  # Desativa log verboso do solver

  Ex = model.addVars(dias, exercicios, slots, vtype=GRB.BINARY, name="Ex")
  Int = model.addVars(
      dias, exercicios, slots, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="Int"
  )
  Fad = model.addVars(
      dias, musculos, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="Fad"
  )
  FadTr = model.addVars(
      dias, slots, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="FadTr"
  )

  # =========================================================================
  # FUNÇÃO OBJETIVO
  # =========================================================================
  model.setObjective(
      gp.quicksum(
          w_slot[k] * utilidade_base[j] * Int[i, j, k]
          for i in dias
          for j in exercicios
          for k in slots
      ),
      GRB.MAXIMIZE,
  )

  # =========================================================================
  # RESTRIÇÕES
  # =========================================================================
  for i in dias:
    # A1. Disponibilidade e exclusividade por slot k
    for k in slots:
      model.addConstr(
          gp.quicksum(Ex[i, j, k] for j in exercicios) <= disp[i],
          name=f"DispSlot_d{i}_k{k}",
      )

    # A2. Preenchimento contíguo (sem buracos na ordem k)
    for idx_k in range(len(slots) - 1):
      k_atual, k_prox = slots[idx_k], slots[idx_k + 1]
      model.addConstr(
          gp.quicksum(Ex[i, j, k_prox] for j in exercicios)
          <= gp.quicksum(Ex[i, j, k_atual] for j in exercicios),
          name=f"SemBuraco_d{i}_k{k_atual}",
      )

    # A3. Não repetir o mesmo exercício no mesmo dia
    for j in exercicios:
      model.addConstr(
          gp.quicksum(Ex[i, j, k] for k in slots) <= 1,
          name=f"NaoRepete_d{i}_{j}",
      )

    # B. Acoplamento Exercício-Intensidade Simplificado (entre 0 e 1, sem Big-M)
    for j in exercicios:
      for k in slots:
        model.addConstr(
            Int[i, j, k] >= I_min * Ex[i, j, k], name=f"IntMin_d{i}_{j}_k{k}"
        )
        model.addConstr(
            Int[i, j, k] <= Ex[i, j, k], name=f"IntMax_d{i}_{j}_k{k}"
        )
        model.addConstr(
            Int[i, j, k] <= 1.0 - FadTr[i, k], name=f"QuedaIntra_d{i}_{j}_k{k}"
        )

    # C. Propagação da Fadiga Intra-Treino (FadTr)
    model.addConstr(FadTr[i, slots[0]] == 0.0, name=f"FadTrInit_d{i}")
    for idx_k in range(len(slots) - 1):
      k_atual, k_prox = slots[idx_k], slots[idx_k + 1]
      model.addConstr(
          FadTr[i, k_prox]
          == FadTr[i, k_atual]
          + gp.quicksum(
              (catalogo[j]["gamma"] * I_hat[j]) * Int[i, j, k_atual]
              for j in exercicios
          ),
          name=f"PropFadTr_d{i}_k{k_prox}",
      )

    # D. Capacidade Diária por Grupo Muscular (Formato B)
    for m in musculos:
      model.addConstr(
          gp.quicksum(
              (catalogo[j]["alpha"].get(m, 0.0) * I_hat[j]) * Int[i, j, k]
              for j in exercicios
              for k in slots
          )
          + gp.quicksum(FadTr[i, k] for k in slots)
          <= 1.0 - Fad[i, m],
          name=f"CapMuscular_d{i}_{m}",
      )

  # E. Evolução da Fadiga Inter-Dias (Fad_{i+1, m})
  for m in musculos:
    model.addConstr(Fad[dias[0], m] == 0.0, name=f"FadInit_{m}")
    for idx_i in range(len(dias) - 1):
      i_atual, i_prox = dias[idx_i], dias[idx_i + 1]
      model.addConstr(
          Fad[i_prox, m]
          == a_fad[m]
          * gp.quicksum(
              (catalogo[j]["alpha"].get(m, 0.0) * I_hat[j]) * Int[i_atual, j, k]
              for j in exercicios
              for k in slots
          )
          + b_fad[m] * Fad[i_atual, m],
          name=f"EvolFad_d{i_prox}_{m}",
      )

  # =========================================================================
  # RESOLUÇÃO E IMPRESSÃO DO RELATÓRIO DE TREINO
  # =========================================================================
  model.optimize()

  print("=" * 85)
  print(f"CENÁRIO: {nome_cenario}")
  if model.status == GRB.OPTIMAL:
    print(f"STATUS: Ótimo Encontrado | Score Hipertrofia: {model.objVal:.2f}")
    print("=" * 85)
    for i in dias:
      status = "DISPONÍVEL" if disp[i] == 1 else "INDISPONÍVEL (Descanso)"
      fad_str = (
          ", ".join([
              f"{m}: {Fad[i, m].X*100:.0f}%"
              for m in musculos
              if Fad[i, m].X > 0.01
          ])
          or "0% (100% Descansado)"
      )
      print(f"\n[DIA {i} - {status}] | Fadiga Muscular Inicial: {fad_str}")

      fez_exercicio = False
      for k in slots:
        for j in exercicios:
          if Ex[i, j, k].X > 0.5:
            fez_exercicio = True
            int_val = Int[i, j, k].X
            peso_prescrito = W_max[j] * int_val
            reps_alvo = catalogo[j]["reps_treino"]
            fad_intra = FadTr[i, k].X
            print(
                f"  -> Slot {k}: {j:<24} | {reps_alvo:>2} reps @ "
                f"{peso_prescrito:>5.1f} kg "
                f"(Int: {int_val*100:>5.1f}% de {W_max[j]:.1f}kg) | "
                f"FadIntra: {fad_intra*100:>4.1f}%"
            )
      if not fez_exercicio and disp[i] == 1:
        print("  -> [Dia poupado pelo solver devido à fadiga acumulada]")
    print("\n")
  else:
    print(f"Erro/Infactível. Status Gurobi: {model.status}\n")


# =============================================================================
# MOCK 1: CENÁRIO SIMPLES (2 Exercícios Distintos, 3 Dias, Até 2 Slots/Dia)
# =============================================================================
mock_simples = {
    "dias": [1, 2, 3],
    "slots": [1, 2],
    "disp": {1: 1, 2: 0, 3: 1},  # Treina Dia 1, descansa Dia 2, treina Dia 3
    "musculos": ["Peito", "Quadriceps"],
    "pref": {
        "Quadriceps": 12.0,
        "Peito": 8.0,
    },  # Prioriza perna sobre peito (define a ordem k!)
    "w_slot": {1: 1.0, 2: 0.90},
    "a_fad": {"Peito": 0.60, "Quadriceps": 0.60},
    "b_fad": {"Peito": 0.30, "Quadriceps": 0.30},
    "catalogo": {
        "Agachamento_Livre": {
            "pr_peso": 120.0,
            "pr_reps": 5,  # PR do usuário: 120kg para 5 reps (1RM = 140kg)
            "reps_treino": 8,  # Quer treinar para 8 reps (W_max = 110.5kg)
            "gamma": 0.12,
            "alpha": {"Quadriceps": 1.0},
        },
        "Supino_Reto": {
            "pr_peso": 80.0,
            "pr_reps": 5,  # PR do usuário: 80kg para 5 reps (1RM = 93.3kg)
            "reps_treino": 10,  # Quer treinar para 10 reps (W_max = 70.0kg)
            "gamma": 0.10,
            "alpha": {"Peito": 1.0},
        },
    },
}


# =============================================================================
# MOCK 2: CENÁRIO COMPLEXO (Semana de 7 Dias, 10 Exercícios, Sinergistas e 4 Slots)
# =============================================================================
mock_complexo = {
    "dias": [1, 2, 3, 4, 5, 6, 7],
    "slots": [1, 2, 3, 4],
    "disp": {
        1: 1,
        2: 1,
        3: 0,
        4: 1,
        5: 1,
        6: 1,
        7: 0,
    },  # Quarta (3) e Domingo (7) indisponíveis
    "musculos": [
        "Peito",
        "Costas",
        "Quadriceps",
        "Posterior_Gluteo",
        "Ombro",
        "Biceps",
        "Triceps",
    ],
    "pref": {
        "Quadriceps": 14.0,
        "Posterior_Gluteo": 14.0,
        "Peito": 11.0,
        "Costas": 11.0,
        "Ombro": 8.0,
        "Biceps": 6.0,
        "Triceps": 6.0,
    },
    "w_slot": {1: 1.00, 2: 0.92, 3: 0.85, 4: 0.78},
    "a_fad": {
        m: 0.65
        for m in [
            "Peito",
            "Costas",
            "Quadriceps",
            "Posterior_Gluteo",
            "Ombro",
            "Biceps",
            "Triceps",
        ]
    },
    "b_fad": {
        m: 0.35
        for m in [
            "Peito",
            "Costas",
            "Quadriceps",
            "Posterior_Gluteo",
            "Ombro",
            "Biceps",
            "Triceps",
        ]
    },
    "catalogo": {
        "Agachamento_Livre": {
            "pr_peso": 120.0,
            "pr_reps": 5,
            "reps_treino": 6,
            "gamma": 0.08,
            "alpha": {"Quadriceps": 1.0, "Posterior_Gluteo": 0.5},
        },
        "Terra_Romeno_RDL": {
            "pr_peso": 110.0,
            "pr_reps": 6,
            "reps_treino": 8,
            "gamma": 0.08,
            "alpha": {"Posterior_Gluteo": 1.0, "Costas": 0.3},
        },
        "Cadeira_Extensora": {
            "pr_peso": 75.0,
            "pr_reps": 10,
            "reps_treino": 12,
            "gamma": 0.04,
            "alpha": {"Quadriceps": 1.0},
        },
        "Mesa_Flexora": {
            "pr_peso": 65.0,
            "pr_reps": 10,
            "reps_treino": 12,
            "gamma": 0.04,
            "alpha": {"Posterior_Gluteo": 1.0},
        },
        "Supino_Reto_Barra": {
            "pr_peso": 85.0,
            "pr_reps": 5,
            "reps_treino": 8,
            "gamma": 0.07,
            "alpha": {"Peito": 1.0, "Ombro": 0.4, "Triceps": 0.5},
        },
        "Desenvolvimento_Halter": {
            "pr_peso": 52.0,
            "pr_reps": 8,
            "reps_treino": 10,
            "gamma": 0.06,
            "alpha": {"Ombro": 1.0, "Triceps": 0.4},
        },
        "Triceps_Corda": {
            "pr_peso": 35.0,
            "pr_reps": 10,
            "reps_treino": 12,
            "gamma": 0.03,
            "alpha": {"Triceps": 1.0},
        },
        "Remada_Curvada": {
            "pr_peso": 80.0,
            "pr_reps": 6,
            "reps_treino": 8,
            "gamma": 0.07,
            "alpha": {
                "Costas": 1.0,
                "Biceps": 0.5,
                "Posterior_Gluteo": 0.2,
            },
        },
        "Puxada_Frontal": {
            "pr_peso": 70.0,
            "pr_reps": 8,
            "reps_treino": 10,
            "gamma": 0.05,
            "alpha": {"Costas": 1.0, "Biceps": 0.4},
        },
        "Rosca_Direta_Barra": {
            "pr_peso": 36.0,
            "pr_reps": 8,
            "reps_treino": 12,
            "gamma": 0.03,
            "alpha": {"Biceps": 1.0},
        },
    },
}

if __name__ == "__main__":
  resolver_modelo_treino("MOCK 1 - SIMPLES (2 Exercícios / 3 Dias)", mock_simples)
  resolver_modelo_treino(
      "MOCK 2 - COMPLEXO (10 Exercícios / 7 Dias)", mock_complexo
  )