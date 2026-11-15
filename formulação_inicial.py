import gurobipy as gp
from gurobipy import GRB

def otimizar_treino_com_descanso():
    # 1. Inicializa o modelo no Gurobi
    modelo = gp.Model("Planejamento_Treino_Fadiga_Descanso")
    
    dias = [1, 2, 3, 4, 5] 
    exercicios = ['Supino', 'Agachamento', 'Remada']
    
    # Ativação muscular ganha por 1 série de cada exercício 
    A = {'Supino': 0.05, 'Agachamento': 0.07, 'Remada': 0.04}
    
    # Fadiga gerada por 1 série de cada exercício 
    Custo_Fadiga = {'Supino': 0.06, 'Agachamento': 0.12, 'Remada': 0.05}
    
    # Fator de retenção de fadiga 
    m = 0.5 
    
    # Limites de volume de treino
    max_series_por_exercicio = 5 
    volume_maximo_dia = 20       
    volume_minimo_dia = 10        
    
    # x[j, i]: Número de séries do exercício j no dia i 
    x = modelo.addVars(exercicios, dias, vtype=GRB.INTEGER, lb=0, ub=max_series_por_exercicio, name="Series")
    
    # y[i]: 1 se o dia i for dia de treino, 0 se for dia de descanso 
    y = modelo.addVars(dias, vtype=GRB.BINARY, name="Dia_de_Treino")
    
    # Tr[i]: qualidade do treino no dia i 
    Tr = modelo.addVars(dias, vtype=GRB.CONTINUOUS, lb=0, name="Tr")
    
    # Esforco[i]: Fadiga gerada pelo treino no dia i 
    Esforco = modelo.addVars(dias, vtype=GRB.CONTINUOUS, lb=0, name="Esforco")
    
    # Fadg[i]: Fadiga total acumulada ao final do dia i
    # Dia 0 para representar a fadiga inicial
    dias_fadiga = [0] + dias
    Fadg = modelo.addVars(dias_fadiga, vtype=GRB.CONTINUOUS, lb=0, name="Fadg")
    
    
    # Maximizar a ativação muscular total nos 5 dias (Função Objetivo)
    modelo.setObjective(gp.quicksum(Tr[i] for i in dias), GRB.MAXIMIZE)
    
    
    # RESTRIÇÕES
    
    # A pessoa começa o dia 1 totalmente descansada
    modelo.addConstr(Fadg[0] == 0, name="Fadiga_Inicial")
    
    for i in dias:
        # 1. Definição da Ativação (Tr) do dia i
        modelo.addConstr(Tr[i] == gp.quicksum(A[j] * x[j, i] for j in exercicios), name=f"Def_Tr_{i}")
        
        # 2. Definição do Esforço do dia i
        modelo.addConstr(Esforco[i] == gp.quicksum(Custo_Fadiga[j] * x[j, i] for j in exercicios), name=f"Def_Esforco_{i}")
        
        # 3. O treino de hoje é limitado pela fadiga de ontem
        modelo.addConstr(Tr[i] <= 1 - Fadg[i-1], name=f"Teto_Prontidao_{i}")
        
        # 4. Definição da fadiga no dia i
        modelo.addConstr(Fadg[i] == Esforco[i] + m * Fadg[i-1], name=f"Atualiza_Fadg_{i}")
        
        # Volume máximo de treinos no dia i
        modelo.addConstr(gp.quicksum(x[j, i] for j in exercicios) <= volume_maximo_dia * y[i], name=f"Volume_Maximo_{i}")
        
        # Volume mínimo de treinos no dia i
        modelo.addConstr(gp.quicksum(x[j, i] for j in exercicios) >= volume_minimo_dia * y[i], name=f"Volume_Minimo_{i}")

    # Otimização do modelo
    modelo.setParam('OutputFlag', 0) 
    modelo.optimize()
    
    # Exibição dos resultados
    if modelo.status == GRB.OPTIMAL:
        print("Status: Solução Ótima Encontrada!")
        print(f"Ativação Total Maximizada (Fobj): {modelo.objVal:.2f}\n")
        
        print("--- PLANEAMENTO DA SEMANA ---")
        for i in dias:
            print(f"\n[ DIA {i} ]")
            print(f"  Teto máximo permitido (1 - Fadg_anterior): {1 - Fadg[i-1].X:.2f}")
            
            if y[i].X > 0.5: 
                print(f"  Ativação alcançada (Tr_{i}): {Tr[i].X:.2f}")
                print(f"  Esforço gerado no dia: {Esforco[i].X:.2f}")
                print(f"  Fadiga final do dia (Fadg_{i}): {Fadg[i].X:.2f}")
                print("  Exercícios:")
                for j in exercicios:
                    qtd_series = x[j, i].X
                    if qtd_series > 0:
                        print(f"    -> {j}: {int(qtd_series)} séries")
            else: 
                print(f"  Fadiga final do dia (Fadg_{i}): {Fadg[i].X:.2f}")
                print("    -> DESCANSO TOTAL (Nenhum exercício realizado)")
    else:
        print("Não foi possível encontrar uma solução ótima.")

# Executar a função
otimizar_treino_com_descanso()
