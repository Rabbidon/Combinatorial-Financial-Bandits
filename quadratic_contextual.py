# Our UCB needs to run an optimiser inside itself
# So we need to split the algo into two bits

# After UCB is implemented we can get some results

# So this is the shell

import numpy as np
import gurobipy as gp
from datetime import datetime

import matplotlib.pyplot as plt

import helper


def UCB(env ,gamma=0, num_features=0, ucb_constant=0):
    
    df = helper.import_preqin_data()
    # We want to maintain a count

    # Now what
    # What would a direct handling look like?
    
    # Ok, we can work out in a theoretical sense what this would correspond to
    # So we are just considering how the residues correlate
    # So we actually just have 2 matrices
    # Feature matrix of each arm
    # And covariance of residuals for each arm
    
    # For each arm we need to store a feature matrix A
    # And a vector b
    
    feature_sums = [np.zeros(num_features+1) for _ in range(len(df.columns)-1)]
    feature_covariances = [np.eye(num_features+1) for _ in range(len(df.columns)-1)]
    
    ucb = np.zeros(len(df.columns)-1)
    cov = np.eye(len(df.columns)-1)
    
    emp_means = np.zeros(len(df.columns)-1)
    num_pulls = np.zeros(len(df.columns)-1)
    
    emp_cov = np.zeros(cov.shape)
    num_cov_pulls = np.zeros(cov.shape)
    
    time = 0
    
    ucb_reward = 0
    
    fama_french = helper.import_fama_french().set_index("Date")
    
    time_series = []
    
    for row in df.iterrows():
        
        month = datetime.strptime(row[1].values[0], '%Y-%m')
        
        # So now we want yoy inflation here
        # Convert datetime to %Y-%m
        
        ff_vector = helper.get_fama_french(fama_french,month)
        
        # Ok, we have a working signal
        # Now what?
        
        # Let's look at the linUCB paper
        
        # Each arm has its own feature mean and covariance matrix
        
        row_data = row[1].values[1:]
        
        # Current feature values at current point in time
        
        feature_vector = np.array(([1]+ff_vector.tolist())[:num_features+1])
        
        time+=1
        print(time)
        l = 0.05

        # Create an empty optimization model
        m = gp.Model(env=env)

        x = m.addMVar(len(df.columns)-1, lb=l, ub=0.3, vtype=gp.GRB.SEMICONT,  name="x")
        b = m.addMVar(len(df.columns)-1, vtype=gp.GRB.BINARY, name="b")
        
        m.addConstr(x.sum() == 1, name="Budget_Constraint")
        m.addConstr(x >= l*b, name="Minimal Position")
        m.addConstr(x <= b, name="Indicator")

        m.addConstr(b.sum() >= 3, "Cardinality")
        m.addConstr(b.sum() <= 10, "Upper Cardinality")
        
        m.setParam('TimeLimit', 1)
    
        # Define objective function: Maximize expected utility
        m.setObjective(
            x@ucb - (gamma / 2) * (x @ cov @ x), gp.GRB.MAXIMIZE
        )
            
        m.optimize()
        m.update()
        
        greedy_allocation = x.X
    
        reward = row_data @ greedy_allocation
        
        # Get nonzero arms
        chosen_arms = [i for i in range(len(df.columns)-1) if greedy_allocation[i]>=l]

        for i in chosen_arms:
            num_pulls[i] += 1
            feature_covariances[i] += np.outer(feature_vector, feature_vector)
            feature_sums[i] += reward*feature_vector
            emp_means[i] += (row_data[i] - emp_means[i])/num_pulls[i]
            for j in chosen_arms:
                num_cov_pulls[i][j] += 1
                emp_cov[i][j] += ((row_data[i]-emp_means[i])*(row_data[j]-emp_means[j]) - emp_cov[i][j])/num_cov_pulls[i][j]

        for i in range(len(df.columns)-1):
            inverse_matrix = np.linalg.inv(feature_covariances[i])
            factor_loading = inverse_matrix @ feature_sums[i]
            ucb[i] = np.dot(factor_loading,feature_vector) + ucb_constant*np.sqrt(feature_vector @ inverse_matrix @ feature_vector)
            for j in range(len(df.columns)-1):
                if num_cov_pulls[i][j]>0:
                    cov[i][j] = emp_cov[i][j] - (1/1+gamma)*ucb_constant*np.sqrt(2 * np.log(time) / num_cov_pulls[i][j])
        
        ucb_reward += reward
        time_series.append(ucb_reward)
    return (time_series,ucb_constant)
        
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

        x = m.addMVar(len(df.columns)-1, lb=l, ub=0.4, vtype=gp.GRB.SEMICONT,  name="x")
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
        
    # Now what?
    # We 
        
    # And what do we do after this?
    # We can see how this performs versus the naive greedy strategy
    # This is more just a proof of concept