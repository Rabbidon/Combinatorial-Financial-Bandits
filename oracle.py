import gurobipy as gp

import dataloader

def optimise(row):

    m = gp.Model()

    x = m.addMVar(len(row), lb=0, ub=0.3, name="x")
    b = m.addMVar(len(row), vtype=gp.GRB.BINARY, name="b")
    
    m.addConstr(x.sum() <= 1, name="Budget_Constraint")
    m.addConstr(x <= b, name="Indicator")

    m.addConstr(b.sum() >= 5, "Cardinality")
    
    m.setObjective(x @ row, gp.GRB.MAXIMIZE)
    
    m.optimize()
    m.update()
    
    print(f"Maximum Reward:     {m.ObjVal:.6f}")
    print(f"Solution time:    {m.Runtime:.2f} seconds\n")
    return (x.X)
    
    # Return the asset allocation