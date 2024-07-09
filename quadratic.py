# Our UCB needs to run an optimiser inside itself
# So we need to split the algo into two bits

# After UCB is implemented we can get some results

# So this is the shell

import numpy as np

import dataloader

import gurobipy as gp

with gp.Env(empty=True) as env:
    
    env.setParam('OutputFlag', 0)
    env.start()

    def UCB(constant=1e-3):
        
        df = dataloader.import_hf_data()
        # We want to maintain a count
        
        ucb = np.ones(len(df.columns)-1)
        cov = np.eye(len(df.columns)-1)
        emp_means = np.zeros(len(df.columns)-1)
        num_pulls = np.zeros(len(df.columns)-1)
        emp_cov = np.zeros(cov.shape)
        num_cov_pulls = np.zeros(cov.shape)
        
        time = 0
        
        ucb_reward = 0
        
        for row in df.iterrows():
            
            row_data = row[1].values[1:]
            
            time+=1
            l = 0.1

            gamma = 0.2  # risk-aversion coefficient

            # Create an empty optimization model
            m = gp.Model()

            x = m.addMVar(len(df.columns)-1, lb=l, ub=0.3, vtype=gp.GRB.SEMICONT,  name="x")
            b = m.addMVar(len(df.columns)-1, vtype=gp.GRB.BINARY, name="b")
            
            m.addConstr(x.sum() == 1, name="Budget_Constraint")
            m.addConstr(x >= l*b, name="Minimal Position")
            m.addConstr(x <= b, name="Indicator")

            m.addConstr(b.sum() >= 5, "Cardinality")
            
            m.setObjective(x @ ucb, gp.GRB.MAXIMIZE)
        
            m.optimize()
            m.update()

            # Define objective function: Maximize expected utility
            m.setObjective(
                x@ucb - (gamma / 2) * (x @ cov @ x), gp.GRB.MAXIMIZE
)
            
            greedy_allocation = x.X
        
            reward = row_data @ greedy_allocation
            
            # Get nonzero arms
            chosen_arms = [i for i in range(len(df.columns)-1) if greedy_allocation[i]>=l]

            for i in chosen_arms:
                num_pulls[i] += 1
                emp_means[i] += (row_data[i]* - emp_means[i])/num_pulls[i]
                for j in chosen_arms:
                    num_cov_pulls[i][j]+=1
                    emp_cov[i][j] += ((row_data[i]-emp_means[i])*(row_data[j]-emp_means[j]) - emp_cov[i][j])/num_cov_pulls[i][j]

            for i in range(len(df.columns)-1):
                if num_pulls[i]>0:
                    ucb[i] = emp_means[i] + constant*np.sqrt(2 * np.log(time) / num_pulls[i])
                for j in range(len(df.columns)-1):
                    if num_cov_pulls[i][j]>0:
                        cov[i][j] = emp_cov[i][j] - constant*np.sqrt(2 * np.log(time) / num_cov_pulls[i][j])
            
            ucb_reward += reward
        return (ucb_reward,constant)
        
    def random():
        
        df = dataloader.import_hf_data()
        reward = 0
        
        for row in df.iterrows():
            
            row_data = row[1].values[1:]
            
            reward+=np.mean(row_data)
        
        return reward
    
    def oracle():
    
        df = dataloader.import_hf_data()
        total_reward = 0
        
        for row in df.iterrows():
            
            row_data = row[1].values[1:]
            
            # generate bernoulli reward from the picked greedy arm
            
            m = gp.Model(env=env)
            l = 0.1

            x = m.addMVar(len(df.columns)-1, lb=l, ub=0.3, vtype=gp.GRB.SEMICONT,  name="x")
            b = m.addMVar(len(df.columns)-1, vtype=gp.GRB.BINARY, name="b")
            
            m.addConstr((x*b).sum() == 1, name="Budget_Constraint")
            m.addConstr(x >= l*b, name="Indicator")

            m.addConstr(b.sum() >= 5, "Cardinality")
            
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

    print(ucb)
    print(greedy)
    print(random_reward)
    print(oracle_reward)
            
        
    # Now what?
    # We 
        
    # And what do we do after this?
    # We can see how this performs versus the naive greedy strategy
    # This is more just a proof of concept