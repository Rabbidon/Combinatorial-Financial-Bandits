# Our UCB needs to run an optimiser inside itself
# So we need to split the algo into two bits

# After UCB is implemented we can get some results

# So this is the shell

import numpy as np

import helper

import matplotlib.pyplot as plt
import gurobipy as gp

with gp.Env(empty=True) as env:
    
    env.setParam('OutputFlag', 0)
    env.start()

    def UCB(constant=3e-3):
        
        df = helper.import_preqin_data()
        # We want to maintain a count
        
        ucb = np.ones(len(df.columns)-1)
        emp_means = np.zeros(len(df.columns)-1)
        num_pulls = np.zeros(len(df.columns)-1)
        
        time = 0
        
        ucb_reward = 0
        time_series = []
        
        for row in df.iterrows():
            
            row_data = row[1].values[1:]
            
            time+=1
            print(time)
            # generate bernoulli reward from the picked greedy arm
            
            m = gp.Model(env=env)
            l = 0.05

            x = m.addMVar(len(df.columns)-1, lb=l, ub=0.4, vtype=gp.GRB.SEMICONT,  name="x")
            b = m.addMVar(len(df.columns)-1, vtype=gp.GRB.BINARY, name="b")
            
            m.addConstr(x.sum() == 1, name="Budget_Constraint")
            m.addConstr(x >= l*b, name="Minimal Position")
            m.addConstr(x <= b, name="Indicator")

            m.addConstr(b.sum() >= 3, "Cardinality")
            m.addConstr(b.sum() <= 10, "Upper Cardinality")
            
            m.setObjective(x @ ucb, gp.GRB.MAXIMIZE)
        
            m.optimize()
            m.update()
            
            greedy_allocation = x.X
            
            reward = row_data @ greedy_allocation
            
            for i in range(len(df.columns)-1):
                if greedy_allocation[i]>0:
                    num_pulls[i] += 1
                    emp_means[i] += (row_data[i] - emp_means[i])/num_pulls[i]
                if num_pulls[i]>0:
                    ucb[i] = emp_means[i] + constant*np.sqrt(2 * np.log(time) / num_pulls[i])
            
            ucb_reward += reward
            time_series.append(ucb_reward)
        return (time_series,constant)
        
    def random():
        
        df = helper.import_preqin_data()
        reward = 0
        
        for row in df.iterrows():
            
            row_data = row[1].values[1:]
            
            reward+=np.mean(row_data)
        
        return reward
    
    def oracle():
    
        df = helper.import_preqin_data()
        total_reward = 0
        
        for row in df.iterrows():
            
            row_data = row[1].values[1:]
            
            # generate bernoulli reward from the picked greedy arm
            
            m = gp.Model(env=env)
            l = 0.05

            x = m.addMVar(len(df.columns)-1, lb=l, ub=0.3, vtype=gp.GRB.SEMICONT,  name="x")
            b = m.addMVar(len(df.columns)-1, vtype=gp.GRB.BINARY, name="b")
            
            m.addConstr((x*b).sum() == 1, name="Budget_Constraint")
            m.addConstr(x >= l*b, name="Indicator")

            m.addConstr(b.sum() >= 3, "Cardinality")
            m.addConstr(b.sum() <= 10, "Upper Cardinality")
            
            m.setObjective((x*b) @ row_data, gp.GRB.MAXIMIZE)
        
            m.optimize()
            m.update()
            
            greedy_allocation = x.X*b.X
            
            reward = row_data @ greedy_allocation
            
            total_reward += reward
        return (total_reward)
    
            
    ucb = UCB()
    greedy = UCB(0)
    random_reward = random()
    oracle_reward = oracle()
    
    print(ucb[0][-1])
    print(greedy[0][-1])
    
    plt.plot(range(len(ucb[0])),ucb[0])
    plt.plot(range(len(ucb[0])),greedy[0])  
    plt.show()
            
        
    # Now what?
    # We 
        
    # And what do we do after this?
    # We can see how this performs versus the naive greedy strategy
    # This is more just a proof of concept