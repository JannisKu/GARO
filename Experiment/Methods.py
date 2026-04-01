import numpy as np
import numpy.typing as npt
import math
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.cluster import KMeans
import gurobipy as gp
from gurobipy import GRB
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import ortho_group

import pandas_datareader.data as web
import pandas as pd 


class OptimizationInstance:

    def __init__(self,n: int, constraint_matrix: npt.NDArray[np.float64], rhs: npt.NDArray[np.float64], sense: str, var_lb: npt.NDArray[np.float64], var_ub: npt.NDArray[np.float64], integer: bool, maximization: bool) -> None:
        self.n = n
        self.A = constraint_matrix
        self.b = rhs
        self.sense = sense
        self.lb = var_lb
        self.ub = var_ub
        self.integer_vars = integer
        self.maximization = maximization
        
        

class Ellipsoid:
    def __init__(self,dataset: npt.NDArray[np.float64], confidence_level: float) -> None:
        self.n = dataset.shape[1]
        self.m = dataset.shape[0]
        self.confidence_level = confidence_level
        self.mean = np.mean(dataset,axis=0)
        self.dataset = dataset
        
        #If eigenvalue is zero add epsilon to diagonal of covariance matrix
        cov_matrix = np.cov(dataset,rowvar = False)
        eigenval, eigenvec = np.linalg.eigh(cov_matrix)
        if abs(eigenval[0])<0.0001:
            print("Added epsilon to diagonal, since eigenvalue too close to zero.")
            cov_matrix = cov_matrix + np.diag(0.0001*np.ones(self.n))
            
        self.matrix = cov_matrix
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.matrix)
        
        self.inv_matrix = np.linalg.inv(self.matrix)
        
        self.gamma = self.getGamma(confidence_level)
        
        
    def __contains__(self,x: npt.NDArray[np.float64]) -> bool:
        vec = np.dot(self.inv_matrix,x-self.mean)
        val = np.dot(x-self.mean,vec)
        
        return val<=self.gamma
    
    def getGamma(self,confidence_level):
        vals = []
        for i in range(self.dataset.shape[0]):
            vec = np.dot(self.inv_matrix,self.dataset[i]-self.mean)
            val = np.dot(self.dataset[i]-self.mean,vec)
            vals.append(val)
            
            
        gamma = np.quantile(vals,confidence_level)
        
        return gamma
    

def clusterData(num_clusters: int, dataset: npt.NDArray[np.float64]) -> list[npt.NDArray[np.float64]]:
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto").fit(dataset)
    
    clusters = []
    
    labels = kmeans.labels_
    
    for i in range(num_clusters):
        data_new = dataset[labels == i]
        clusters.append(data_new[:])
        
    return clusters

def runRobustOptimization(dataset: npt.NDArray[np.float64], num_clusters: int, confidence_level: float, opt_inst: OptimizationInstance, uncert_indices: list, non_neg_scen: bool = False) -> npt.NDArray[np.float64] | float:
    
    clusters = clusterData(num_clusters = num_clusters, dataset = dataset)
    n = opt_inst.n
    m = opt_inst.A.shape[0]
    q = len(uncert_indices)
    
    ellipsoids = []
    for U in clusters:
        new_ellips = Ellipsoid(dataset = U[:], confidence_level = confidence_level)
        ellipsoids.append(new_ellips)
        
       
    
    # start algorithm
    model = gp.Model("Classical_RO")
    model.setParam("OutputFlag",0)
    model.setParam('OptimalityTol', 1e-4)  

    # Create variables

    z = model.addVar(lb = -GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z")
    sig = model.addVars(num_clusters,vtype=GRB.CONTINUOUS, name = "sig")
    
    if opt_inst.integer_vars:
        x = model.addVars(n, lb = opt_inst.lb, ub = opt_inst.ub, vtype=GRB.INTEGER, name = "x")
    else:
        x = model.addVars(n, lb = opt_inst.lb, ub = opt_inst.ub, vtype=GRB.CONTINUOUS, name = "x")
        
    if confidence_level == 0:
        non_neg_scen = False
        
    if non_neg_scen:
        nu = model.addVars(q, vtype=GRB.CONTINUOUS, name = "nu")
        

    if opt_inst.maximization:
        model.setObjective(1 * z, GRB.MAXIMIZE)
    else:
        model.setObjective(1 * z, GRB.MINIMIZE)
        
        
    #Constraints of Optimization Instance    
    for i in range(m):
        if opt_inst.sense == "<":
            model.addConstr(gp.quicksum(opt_inst.A[i,j]*x[j] for j in range(n)) <= opt_inst.b[i], name="opt_instance_constraint")
        elif opt_inst.sense == ">":
            model.addConstr(gp.quicksum(opt_inst.A[i,j]*x[j] for j in range(n)) >= opt_inst.b[i], name="opt_instance_constraint")
        elif opt_inst.sense == "=":
            model.addConstr(gp.quicksum(opt_inst.A[i,j]*x[j] for j in range(n)) == opt_inst.b[i], name="opt_instance_constraint")
    
        
    #Modelling the squareroot with new variable sig[k]
    for k in range(len(ellipsoids)):
        if non_neg_scen:
            if opt_inst.maximization:
                model.addQConstr(gp.quicksum(ellipsoids[k].matrix[i,j]*(x[uncert_indices[i]]+nu[i]) * (x[uncert_indices[j]]+nu[j]) for i in range(q) for j in range(q)) - sig[k]*sig[k] <= 0, name="sqrt_constraint")
            else:
                model.addQConstr(gp.quicksum(ellipsoids[k].matrix[i,j]*(x[uncert_indices[i]]-nu[i]) * (x[uncert_indices[j]]-nu[j]) for i in range(q) for j in range(q)) - sig[k]*sig[k] <= 0, name="sqrt_constraint")
        else:
            model.addQConstr(gp.quicksum(ellipsoids[k].matrix[i,j]*x[uncert_indices[i]] * x[uncert_indices[j]] for i in range(q) for j in range(q)) - sig[k]*sig[k] <= 0, name="sqrt_constraint")
        
        
    #Modelling the maximization over ellipsoid
    for k in range(len(ellipsoids)):
        if non_neg_scen:
            if opt_inst.maximization:
                model.addConstr(gp.quicksum(ellipsoids[k].mean[j]*(x[uncert_indices[j]]+nu[j]) for j in range(q)) - math.sqrt(ellipsoids[k].gamma)*sig[k] - z >= 0, name="ellips_constraint")
            else:
                model.addConstr(gp.quicksum(ellipsoids[k].mean[j]*(x[uncert_indices[j]]-nu[j]) for j in range(q)) + math.sqrt(ellipsoids[k].gamma)*sig[k] - z <= 0, name="ellips_constraint")
        else:
            if opt_inst.maximization:
                model.addConstr(gp.quicksum(ellipsoids[k].mean[j]*x[uncert_indices[j]] for j in range(q)) - math.sqrt(ellipsoids[k].gamma)*sig[k] - z >= 0, name="ellips_constraint")
            else:
                model.addConstr(gp.quicksum(ellipsoids[k].mean[j]*x[uncert_indices[j]] for j in range(q)) + math.sqrt(ellipsoids[k].gamma)*sig[k] - z <= 0, name="ellips_constraint")
        
    print("Solve Classic RO Problem. Confidence=",confidence_level, "Gamma:",ellipsoids[0].gamma)
    model.optimize()
    model.write("Classical_RO.lp")
    
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found.")
        x_opt = np.array(list(model.getAttr("X",x).values()))
        obj = model.objVal
    elif model.status == GRB.INFEASIBLE:
        print("Model is infeasible.")
    elif model.status == GRB.UNBOUNDED:
        print("Model is unbounded.")
    elif model.status == GRB.INF_OR_UNBD:
        print("Model is infeasible or unbounded.")
    elif model.status == GRB.SUBOPTIMAL:
        print("Model unable to satisfy optimality tolerances")
        x_opt = np.array(list(model.getAttr("X",x).values()))
        obj = model.objVal
        print("Alpha:",obj)
    else:
        print(f"Optimization ended with status {model.status}")
        
    model.dispose()
    
    
    return x_opt, obj

def minMaxEllipsoid(ellipsoid: Ellipsoid, opt_inst: OptimizationInstance, uncert_indices: list, non_neg_scen: bool = False) -> npt.NDArray[np.float64] | float:
    
    n = opt_inst.n
    m = opt_inst.A.shape[0]
    q = len(uncert_indices)
    
    # start algorithm

    model = gp.Model("MinMaxEllipsoid")
    model.setParam("OutputFlag",0)
    model.setParam('OptimalityTol', 1e-4)
    # Create variables

    z = model.addVar(lb = -GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z")
    sig = model.addVar(vtype=GRB.CONTINUOUS, name = "sig")
    
    if opt_inst.integer_vars:
        x = model.addVars(n, lb = opt_inst.lb, ub = opt_inst.ub, vtype=GRB.INTEGER, name = "x")
    else:
        x = model.addVars(n, lb = opt_inst.lb, ub = opt_inst.ub, vtype=GRB.CONTINUOUS, name = "x")
    
        
    if ellipsoid.gamma == 0:
        non_neg_scen = False
    if non_neg_scen:
        nu = model.addVars(q, vtype=GRB.CONTINUOUS, name = "nu")
        
    
    if opt_inst.maximization:
        model.setObjective(1 * z, GRB.MAXIMIZE)
    else:
        model.setObjective(1 * z, GRB.MINIMIZE)
        
        
    #Constraints of Optimization Instance    
    for i in range(m):
        if opt_inst.sense == "<":
            model.addConstr(gp.quicksum(opt_inst.A[i,j]*x[j] for j in range(n)) <= opt_inst.b[i], name="opt_instance_constraint")
        elif opt_inst.sense == ">":
            model.addConstr(gp.quicksum(opt_inst.A[i,j]*x[j] for j in range(n)) >= opt_inst.b[i], name="opt_instance_constraint")
        elif opt_inst.sense == "=":
            model.addConstr(gp.quicksum(opt_inst.A[i,j]*x[j] for j in range(n)) == opt_inst.b[i], name="opt_instance_constraint")
    
        
    #Modelling the squareroot with new variable sig[k]

    if non_neg_scen:
        if opt_inst.maximization:
            model.addQConstr(gp.quicksum(ellipsoid.matrix[i,j]*(x[uncert_indices[i]]+nu[i]) * (x[uncert_indices[j]]+nu[j]) for i in range(q) for j in range(q)) - sig*sig <= 0, name="sqrt_constraint")
        else:
            model.addQConstr(gp.quicksum(ellipsoid.matrix[i,j]*(x[uncert_indices[i]]-nu[i]) * (x[uncert_indices[j]]-nu[j]) for i in range(q) for j in range(q)) - sig*sig <= 0, name="sqrt_constraint")
    else:
        model.addQConstr(gp.quicksum(ellipsoid.matrix[i,j]*x[uncert_indices[i]] * x[uncert_indices[j]] for i in range(q) for j in range(q)) - sig*sig <= 0, name="sqrt_constraint")
    
        
    #Modelling the maximization over ellipsoid

    if non_neg_scen:
        if opt_inst.maximization:
            model.addConstr(gp.quicksum(ellipsoid.mean[j]*(x[uncert_indices[j]]+nu[j]) for j in range(q)) - math.sqrt(ellipsoid.gamma)*sig - z >= 0, name="ellips_constraint")
        else:
            model.addConstr(gp.quicksum(ellipsoid.mean[j]*(x[uncert_indices[j]]-nu[j]) for j in range(q)) + math.sqrt(ellipsoid.gamma)*sig - z <= 0, name="ellips_constraint")
    else:
        if opt_inst.maximization:
            model.addConstr(gp.quicksum(ellipsoid.mean[j]*x[uncert_indices[j]] for j in range(q)) - math.sqrt(ellipsoid.gamma)*sig - z >= 0, name="ellips_constraint")
        else:
            model.addConstr(gp.quicksum(ellipsoid.mean[j]*x[uncert_indices[j]] for j in range(q)) + math.sqrt(ellipsoid.gamma)*sig - z <= 0, name="ellips_constraint")
    
    print("Solve Classic RO Problem. Gamma=",ellipsoid.gamma)
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found.")
        x_opt = np.array(list(model.getAttr("X",x).values()))
        obj = model.objVal
    elif model.status == GRB.INFEASIBLE:
        print("Model is infeasible.")
    elif model.status == GRB.UNBOUNDED:
        print("Model is unbounded.")
    elif model.status == GRB.INF_OR_UNBD:
        print("Model is infeasible or unbounded.")
    elif model.status == GRB.SUBOPTIMAL:
        print("Model unable to satisfy optimality tolerances")
        x_opt = np.array(list(model.getAttr("X",x).values()))
        obj = model.objVal
        print("Alpha:",obj)
    else:
        print(f"Optimization ended with status {model.status}")
        
    model.dispose()
    
    return x_opt, obj


def runRobustOptimizationScenarios(dataset: npt.NDArray[np.float64], num_clusters: int, confidence_level: float, opt_inst: OptimizationInstance, uncert_indices: list) -> npt.NDArray[np.float64] | float:
    
    clusters = clusterData(num_clusters = num_clusters, dataset = dataset)
    n = opt_inst.n
    m = opt_inst.A.shape[0]
    q = len(uncert_indices)
    
    ellipsoids = []
    for U in clusters:
        new_ellips = Ellipsoid(dataset = U, confidence_level = confidence_level)
        ellipsoids.append(new_ellips)
        
       
    # start algorithm
    

    model = gp.Model("Classical_RO_Scenarios")
    model.setParam("OutputFlag",0)
    model.setParam('OptimalityTol', 1e-4)
    # Create variables

    z = model.addVar(lb = -GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z")
    
    if opt_inst.integer_vars:
        x = model.addVars(n, lb = opt_inst.lb, ub = opt_inst.ub, vtype=GRB.INTEGER, name = "x")
    else:
        x = model.addVars(n, lb = opt_inst.lb, ub = opt_inst.ub, vtype=GRB.CONTINUOUS, name = "x")
        

    if opt_inst.maximization:
        model.setObjective(1 * z, GRB.MAXIMIZE)
    else:
        model.setObjective(1 * z, GRB.MINIMIZE)
        
        
    #Constraints of Optimization Instance    
    for i in range(m):
        if opt_inst.sense == "<":
            model.addConstr(gp.quicksum(opt_inst.A[i,j]*x[j] for j in range(n)) <= opt_inst.b[i], name="opt_instance_constraint")
        elif opt_inst.sense == ">":
            model.addConstr(gp.quicksum(opt_inst.A[i,j]*x[j] for j in range(n)) >= opt_inst.b[i], name="opt_instance_constraint")
        elif opt_inst.sense == "=":
            model.addConstr(gp.quicksum(opt_inst.A[i,j]*x[j] for j in range(n)) == opt_inst.b[i], name="opt_instance_constraint")
    
    #Add Mean Scenario
    for k in range(len(ellipsoids)):
        if opt_inst.maximization:
            model.addConstr(gp.quicksum(ellipsoids[k].mean[j]*x[uncert_indices[j]] for j in range(q)) - z >= 0, name="scenario_constraint")
        else:
           model.addConstr(gp.quicksum(ellipsoids[k].mean[j]*x[uncert_indices[j]] for j in range(q)) - z <= 0, name="scenario_constraint")
           
        
    #Modelling the scenario constraints
    for k in range(len(ellipsoids)):
        counter = 0
        for scenario in clusters[k]:
            if scenario in ellipsoids[k]:
                counter+=1
                if opt_inst.maximization:
                    model.addConstr(gp.quicksum(scenario[j]*x[uncert_indices[j]] for j in range(q)) - z >= 0, name="scenario_constraint")
                else:
                   model.addConstr(gp.quicksum(scenario[j]*x[uncert_indices[j]] for j in range(q)) - z <= 0, name="scenario_constraint")
    
        print("Confidence:", confidence_level, "Num Samples in Cluster:", len(clusters[k]), "Fraction Samples:", counter/len(clusters[k]))
       
    print("Solve Classic Scenario RO Problem. Confidence=",confidence_level)
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found.")
        x_opt = np.array(list(model.getAttr("X",x).values()))
        obj = model.objVal
    elif model.status == GRB.INFEASIBLE:
        print("Model is infeasible.")
    elif model.status == GRB.UNBOUNDED:
        print("Model is unbounded.")
    elif model.status == GRB.INF_OR_UNBD:
        print("Model is infeasible or unbounded.")
    elif model.status == GRB.SUBOPTIMAL:
        print("Model unable to satisfy optimality tolerances")
        x_opt = np.array(list(model.getAttr("X",x).values()))
        obj = model.objVal
        print("Alpha:",obj)
    else:
        print(f"Optimization ended with status {model.status}")
        
    model.dispose()
    
    
    return x_opt, obj



def runClassicalRegret(dataset: npt.NDArray[np.float64], num_clusters: int, gamma_val: float, opt_inst: OptimizationInstance, uncert_indices: list) -> npt.NDArray[np.float64] | float:
    
    n = opt_inst.n
    m = opt_inst.A.shape[0]
    q = len(uncert_indices)
    

    ellipsoid = Ellipsoid(dataset = dataset, confidence_level = 0.99)
    ellipsoid.gamma = gamma_val

    # start algorithm
    

    model = gp.Model("Classical_Regret")
    model.setParam("OutputFlag",0)
    model.setParam('OptimalityTol', 1e-4)
    
    # Create variables

    z = model.addVar(lb = -GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z")
    
    if opt_inst.integer_vars:
        x = model.addVars(n, lb = opt_inst.lb, ub = opt_inst.ub, vtype=GRB.INTEGER, name = "x")
    else:
        x = model.addVars(n, lb = opt_inst.lb, ub = opt_inst.ub, vtype=GRB.CONTINUOUS, name = "x")

    model.setObjective(1 * z, GRB.MINIMIZE)
        
    #Constraints of Optimization Instance    
    for i in range(m):
        if opt_inst.sense == "<":
            model.addConstr(gp.quicksum(opt_inst.A[i,j]*x[j] for j in range(n)) <= opt_inst.b[i], name="opt_instance_constraint")
        elif opt_inst.sense == ">":
            model.addConstr(gp.quicksum(opt_inst.A[i,j]*x[j] for j in range(n)) >= opt_inst.b[i], name="opt_instance_constraint")
        elif opt_inst.sense == "=":
            model.addConstr(gp.quicksum(opt_inst.A[i,j]*x[j] for j in range(n)) == opt_inst.b[i], name="opt_instance_constraint")
    
    #Add Mean Scenario
    objective_vector = np.zeros(n)
    objective_vector[uncert_indices] = ellipsoid.mean
    x_det, val_det = runDeterministicProblem(c = objective_vector,opt_inst = opt_inst)
    
    if opt_inst.maximization:
        model.addConstr(gp.quicksum(-ellipsoid.mean[j]*x[uncert_indices[j]] for j in range(q)) - z <= -val_det, name="scenario_constraint")
    else:
        model.addConstr(gp.quicksum(ellipsoid.mean[j]*x[uncert_indices[j]] for j in range(q)) - z <= val_det, name="scenario_constraint")
        
    #Modelling the scenario constraints

    counter = 0
    for scenario in dataset:
        if scenario in ellipsoid:
            counter+=1
            objective_vector = np.zeros(n)
            objective_vector[uncert_indices] = scenario
            x_det, val_det = runDeterministicProblem(c = objective_vector,opt_inst = opt_inst)
            
            if opt_inst.maximization:
                model.addConstr(gp.quicksum(-scenario[j]*x[uncert_indices[j]] for j in range(q)) - z <= -val_det, name="scenario_constraint")
            else:
                model.addConstr(gp.quicksum(scenario[j]*x[uncert_indices[j]] for j in range(q)) - z <= val_det, name="scenario_constraint")


    print("Solve Classic Regret Problem. Gamma=", gamma_val)
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found.")
        x_opt = np.array(list(model.getAttr("X",x).values()))
        obj = model.objVal
    elif model.status == GRB.INFEASIBLE:
        print("Model is infeasible.")
    elif model.status == GRB.UNBOUNDED:
        print("Model is unbounded.")
    elif model.status == GRB.INF_OR_UNBD:
        print("Model is infeasible or unbounded.")
    elif model.status == GRB.SUBOPTIMAL:
        print("Model unable to satisfy optimality tolerances")
        x_opt = np.array(list(model.getAttr("X",x).values()))
        obj = model.objVal
        print("Alpha:",obj)
    else:
        print(f"Optimization ended with status {model.status}")
        
    model.dispose()
    
    
    return x_opt, obj


def runDeterministicProblem(c: npt.NDArray[np.float64],opt_inst: OptimizationInstance) -> npt.NDArray[np.float64] | float:
    n = opt_inst.n
    m = opt_inst.A.shape[0]
    
    model = gp.Model("Deterministic_Problem")
    model.setParam("OutputFlag",0)
    # Create variables
    
    if opt_inst.integer_vars:
        x = model.addVars(n, lb = opt_inst.lb, ub = opt_inst.ub, vtype=GRB.INTEGER, name = "x")
    else:
        x = model.addVars(n, lb = opt_inst.lb, ub = opt_inst.ub, vtype=GRB.CONTINUOUS, name = "x")


    if opt_inst.maximization:
        model.setObjective(gp.quicksum(c[j]*x[j] for j in range(n)), GRB.MAXIMIZE)
    else:
        model.setObjective(gp.quicksum(c[j]*x[j] for j in range(n)), GRB.MINIMIZE)
        
        
    #Constraints of Optimization Instance    
    for i in range(m):
        if opt_inst.sense == "<":
            model.addConstr(gp.quicksum(opt_inst.A[i,j]*x[j] for j in range(n)) <= opt_inst.b[i], name="opt_instance_constraint")
        elif opt_inst.sense == ">":
            model.addConstr(gp.quicksum(opt_inst.A[i,j]*x[j] for j in range(n)) >= opt_inst.b[i], name="opt_instance_constraint")
        elif opt_inst.sense == "=":
            model.addConstr(gp.quicksum(opt_inst.A[i,j]*x[j] for j in range(n)) == opt_inst.b[i], name="opt_instance_constraint")
    

    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        x_opt = np.array(list(model.getAttr("X",x).values()))
        obj = model.objVal
    elif model.status == GRB.INFEASIBLE:
        print("Model is infeasible.")
    elif model.status == GRB.UNBOUNDED:
        print("Model is unbounded.")
    elif model.status == GRB.INF_OR_UNBD:
        print("Model is infeasible or unbounded.")
    else:
        print(f"Optimization ended with status {model.status}")
        
    model.dispose()
        
    return x_opt, obj
    


def runGARO(dataset: npt.NDArray[np.float64], p: float, opt_inst: OptimizationInstance, uncert_indices: list) -> npt.NDArray[np.float64] | float:
    print("Solve Robust Global Regret Problem: p=",p)
    
    #uncert_indices works for the case where the variables with uncertain coefficients in the objective function (uncert_indices) are the only ones
    #appearing in the objective function and all other variables are only appearing in the constraints.
    
    n = opt_inst.n
    m = opt_inst.A.shape[0]
    q = len(uncert_indices)
    
    ellipsoid = Ellipsoid(dataset = dataset, confidence_level = 0.99)
    
    Gamma = ellipsoid.gamma
    gamma_list = np.linspace(0,Gamma,100)
    minmax_values = []
    
    for g in gamma_list:
        ellipsoid.gamma = g
        x_dummy, val = minMaxEllipsoid(ellipsoid, opt_inst, uncert_indices)

        minmax_values.append(val)
    
       
    # start algorithm
    
    eps = 0.0005
    model = gp.Model("Global_Regret_RO")
    model.setParam("OutputFlag",0)
    model.setParam('OptimalityTol', 1e-4)
    # Create variables

    alpha = model.addVar(vtype=GRB.CONTINUOUS, name="alpha")
    sig = model.addVar(vtype=GRB.CONTINUOUS, name = "sig")
    
    if opt_inst.integer_vars:
        x = model.addVars(n, lb = opt_inst.lb, ub = opt_inst.ub, vtype=GRB.INTEGER, name = "x")
    else:
        x = model.addVars(n, lb = opt_inst.lb, ub = opt_inst.ub, vtype=GRB.CONTINUOUS, name = "x")

    model.setObjective(1 * alpha, GRB.MINIMIZE)
        
        
    #Constraints of Optimization Instance    
    for i in range(m):
        if opt_inst.sense == "<":
            model.addConstr(gp.quicksum(opt_inst.A[i,j]*x[j] for j in range(n)) <= opt_inst.b[i], name="opt_instance_constraint")
        elif opt_inst.sense == ">":
            model.addConstr(gp.quicksum(opt_inst.A[i,j]*x[j] for j in range(n)) >= opt_inst.b[i], name="opt_instance_constraint")
        elif opt_inst.sense == "=":
            model.addConstr(gp.quicksum(opt_inst.A[i,j]*x[j] for j in range(n)) == opt_inst.b[i], name="opt_instance_constraint")
    
    
    #Modelling the squareroot with new variable sig

    model.addQConstr(gp.quicksum(ellipsoid.matrix[i,j]*x[uncert_indices[i]] * x[uncert_indices[j]] for i in range(q) for j in range(q)) - sig*sig <= 0, name="sqrt_constraint")
        
    index = -1
    for g in gamma_list:
        index+=1
        if opt_inst.maximization:
            model.addConstr(gp.quicksum(-ellipsoid.mean[j]*x[uncert_indices[j]] for j in range(q)) + math.sqrt(g) * sig - alpha*(g+1)**p\
                        <= -minmax_values[index])
        else:
            model.addConstr(gp.quicksum(ellipsoid.mean[j]*x[uncert_indices[j]] for j in range(q)) + math.sqrt(g) * sig - alpha*(g+1)**p\
                        <= minmax_values[index])
        
    
    model.optimize()
    if model.status == GRB.OPTIMAL:
        x_opt = np.array(list(model.getAttr("X",x).values()))
        alpha_opt =  alpha.x
            
        print("Alpha:",alpha_opt)
    elif model.status == GRB.INFEASIBLE:
        print("Model is infeasible.")
        return None
    elif model.status == GRB.UNBOUNDED:
        print("Model is unbounded.")
        return None
    elif model.status == GRB.INF_OR_UNBD:
        print("Model is infeasible or unbounded.")
        return None
    elif model.status == GRB.SUBOPTIMAL:
        print("Model unable to satisfy optimality tolerances")
        x_opt = np.array(list(model.getAttr("X",x).values()))
        alpha_opt =  alpha.x
        print("Alpha:",alpha_opt)
    else:
        print(f"Optimization ended with status {model.status}")
        return None

        
    return x_opt, alpha_opt



def runRobustSatisficing(dataset: npt.NDArray[np.float64], f_target: float, opt_inst: OptimizationInstance, uncert_indices: list) -> npt.NDArray[np.float64] | float:
    print("Solve Robust Satisficing for: f_target=",f_target)
    
    n = opt_inst.n
    m = opt_inst.A.shape[0]
    q = len(uncert_indices)
    

    ellipsoid = Ellipsoid(dataset = dataset, confidence_level = 0.99)
    
    Gamma = ellipsoid.gamma
    gamma_list = np.linspace(0,Gamma,100)
       
    # start algorithm
    
    eps = 0.0005
    model = gp.Model("Robust Satisficing")
    model.setParam("OutputFlag",0)
    model.setParam('OptimalityTol', 1e-4)  
    
    # Create variables

    alpha = model.addVar(vtype=GRB.CONTINUOUS, name="alpha")
    sig = model.addVar(vtype=GRB.CONTINUOUS, name = "sig")
    
    if opt_inst.integer_vars:
        x = model.addVars(n, lb = opt_inst.lb, ub = opt_inst.ub, vtype=GRB.INTEGER, name = "x")
    else:
        x = model.addVars(n, lb = opt_inst.lb, ub = opt_inst.ub, vtype=GRB.CONTINUOUS, name = "x")

    model.setObjective(1 * alpha, GRB.MINIMIZE)
        
        
    #Constraints of Optimization Instance    
    for i in range(m):
        if opt_inst.sense == "<":
            model.addConstr(gp.quicksum(opt_inst.A[i,j]*x[j] for j in range(n)) <= opt_inst.b[i], name="opt_instance_constraint")
        elif opt_inst.sense == ">":
            model.addConstr(gp.quicksum(opt_inst.A[i,j]*x[j] for j in range(n)) >= opt_inst.b[i], name="opt_instance_constraint")
        elif opt_inst.sense == "=":
            model.addConstr(gp.quicksum(opt_inst.A[i,j]*x[j] for j in range(n)) == opt_inst.b[i], name="opt_instance_constraint")
    
    
    #Modelling the squareroot with new variable sig

    model.addQConstr(gp.quicksum(ellipsoid.matrix[i,j]*x[uncert_indices[i]] * x[uncert_indices[j]] for i in range(q) for j in range(q)) - sig*sig <= 0, name="sqrt_constraint")
        
    index = -1
    for g in gamma_list:
        index+=1
        if opt_inst.maximization:
            model.addConstr(gp.quicksum(-ellipsoid.mean[j]*x[uncert_indices[j]] for j in range(q)) + math.sqrt(g) * sig - alpha*g\
                        <= -f_target)
        else:
            model.addConstr(gp.quicksum(ellipsoid.mean[j]*x[uncert_indices[j]] for j in range(q)) + math.sqrt(g) * sig - alpha*g\
                        <= f_target)
        
    
    model.optimize()
    if model.status == GRB.OPTIMAL:
        x_opt = np.array(list(model.getAttr("X",x).values()))
        alpha_opt =  alpha.x
            
        print("Alpha:",alpha_opt)
    elif model.status == GRB.INFEASIBLE:
        print("Model is infeasible.")
        return None
    elif model.status == GRB.UNBOUNDED:
        print("Model is unbounded.")
        return None
    elif model.status == GRB.INF_OR_UNBD:
        print("Model is infeasible or unbounded.")
        return None
    elif model.status == GRB.SUBOPTIMAL:
        print("Model unable to satisfy optimality tolerances")
        x_opt = np.array(list(model.getAttr("X",x).values()))
        alpha_opt =  alpha.x
        print("Alpha:",alpha_opt)
    else:
        print(f"Optimization ended with status {model.status}")
        return None

        
    return x_opt, alpha_opt



def getSolutionStatistics(x: npt.NDArray[np.float64], test_data: npt.NDArray[np.float64], costs: str = "linear") -> float | float:
    
    if costs == "linear":
        values = np.dot(test_data,x)
    elif costs == "l2":
        values = np.linalg.norm(test_data - x, axis=1)**2
        
    
    avg = np.average(values)
    std = np.std(values)
    
    cov_matrix = np.cov(test_data,rowvar = False)
    vec = np.dot(cov_matrix, x)
    risk = math.sqrt(np.dot(x,vec))
    maxv = np.amax(values) 
    minv = np.amin(values) 
    
    return avg, std, risk, maxv, minv


def generateGaussianData(dim: int, num_points: int, small: bool) -> npt.NDArray[np.float64] | npt.NDArray[np.float64]:
    mean = 25*np.random.rand(dim)
    onb = ortho_group.rvs(dim=dim)
    
    
    sigma = np.power(np.random.rand(dim)*0.25*mean,2*np.ones(dim))

    Sigma = 0
    for i in range(dim):
        Sigma = Sigma + sigma[i]*np.outer(onb[i,:],onb[i,:])

    data = np.random.multivariate_normal(mean, Sigma, num_points)
    
    if small:
        data_train, data_test = train_test_split(data, test_size=0.9, random_state=42)
    else:
        data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)
    
    return data_train, data_test


def generateGaussianContrary(dim: int, num_points: int, small: bool) -> npt.NDArray[np.float64] | npt.NDArray[np.float64]:
    mean = 50*np.random.rand(dim)
    onb = ortho_group.rvs(dim=dim)
    
    
    sigma = np.power(np.random.rand(dim)*(50-mean),2*np.ones(dim))

    Sigma = 0
    for i in range(dim):
        Sigma = Sigma + sigma[i]*np.outer(onb[i,:],onb[i,:])

    data = np.random.multivariate_normal(mean, Sigma, num_points)
    
    if small:
        data_train, data_test = train_test_split(data, test_size=0.9, random_state=42)
    else:
        data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)
    
    return data_train, data_test


def generateHeavyTail(dim: int, num_points: int, small: bool) -> npt.NDArray[np.float64] | npt.NDArray[np.float64]:
    alpha = 1.5  
    scale = 1

    data = (np.random.pareto(alpha, size = (num_points,dim)) + 1) * scale
    
    if small:
        data_train, data_test = train_test_split(data, test_size=0.9, random_state=42)
    else:
        data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)
    
    return data_train, data_test

















