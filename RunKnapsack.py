import Methods as gror
import os
import time
import csv
from datetime import datetime
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from gurobipy import GRB
import pickle

# Run Knapsack Instances 

#####################
##    START PROGRAM   ###
######################

os.environ["OMP_NUM_THREADS"] = "4"

#Setup Parameters
num_points = 5000
dim = 25
integer_vars = False

maximization_prob = False
dataType = "Heavy_Tail"

num_data_instances = 1
num_opt_instances = 1
num_clusters = 1

#setup only RO
#setup_RO = np.round([0.7**i for i in range(21)],3)

setup_RO = np.round([0 + i*0.02 for i in range(5)],3)
setup_RO_d = np.round([0.1 + i*0.1 for i in range(5)],3)
setup_REG = np.round([0.1 + i*0.1 for i in range(5)],3)
setup_GARO = np.round([0 + i*0.5 for i in range(5)],3)
setup_Sat = np.round([1.2 + i*0.2 for i in range(5)],3)


ct = datetime.now()
string_ct = str(ct.year) + "_" + str(ct.month) + "_" + str(ct.day) + "_" + str(ct.hour) + "_" + str(ct.minute)
output_folder = "Experiments\\" + string_ct 
if not os.path.exists(output_folder):
    
    os.makedirs(output_folder)

filename_output = output_folder + "\\Results"



train_sets = []
test_sets = []
for i in range(num_data_instances):
    if dataType == "Gaussian":
        U_train, U_test = gror.generateGaussianData(dim = dim, num_points = num_points, small = False)
    elif dataType == "Gaussian_contrary":
        U_train, U_test = gror.generateGaussianContrary(dim = dim, num_points = num_points, small = False)
    elif dataType == "Heavy_Tail":
        U_train, U_test = gror.generateHeavyTail(dim = dim, num_points = num_points, small = False)
        
    train_sets.append(U_train[:])
    test_sets.append(U_test[:])
    
    df = pd.DataFrame(U_train)
    df.to_csv(output_folder + "\\data_train_" + str(i)+  ".csv",header=False)
    
    df = pd.DataFrame(U_test)
    df.to_csv(output_folder + "\\data_test_" + str(i)+  ".csv",header=False)

opt_instances = []    
for i in range(num_opt_instances):
    a = np.array([25*np.ones(dim) + 75*np.random.rand(dim)],dtype=int)
    print(a)
    b = np.array([0.4*np.sum(a)])
    sense = ">"
    
    lb = np.zeros(dim)
    #ub = np.ones(dim)
    ub = np.ones(dim)*100
    opt_instances.append(gror.OptimizationInstance(n = dim, constraint_matrix = a, rhs = b, sense = sense, var_lb = lb, var_ub = ub, integer = integer_vars, maximization = maximization_prob))

    csv_file = open(output_folder + "\\Instances.csv", 'a')
    out = csv.writer(csv_file,delimiter=";", lineterminator = "\n")
    out.writerow([i])
    out.writerow(a[0,:])
    out.writerow([b])
    out.writerow([sense])
    out.writerow(lb)
    out.writerow(ub)
    out.writerow([integer_vars])
    out.writerow([maximization_prob])
    out.writerow([])
    
    csv_file.close()


gamma_list = np.linspace(0,1,100)
Gamma_max_values = []
minmax_values = []
test_data_values = []

for data_index_guarantee in range(num_data_instances):
    ellipsoid = gror.Ellipsoid(dataset = train_sets[data_index_guarantee], confidence_level = 0.99)
    Gamma_max_values.append(ellipsoid.gamma)
    
Gamma_max_value = max(Gamma_max_values)

#Calculate classical robust values for guarantee-plots
for data_index_guarantee in range(num_data_instances):
    ellipsoid = gror.Ellipsoid(dataset = train_sets[data_index_guarantee], confidence_level = 0.99)

    
    #calculate gamma values for test data
    test_values = []
    for point in test_sets[data_index_guarantee]:
        #calculate gamma value
        vec = np.dot(ellipsoid.inv_matrix,point-ellipsoid.mean)
        point_gamma = np.dot(point-ellipsoid.mean,vec)
        test_values.append(point_gamma)
    
    
    test_values.sort()
    test_values = np.array(test_values) / Gamma_max_value
    test_data_values.append(test_values)
        
        
test_data_values = np.array(test_data_values)
test_data_means = np.mean(test_data_values, axis = 0)

max_gamma_plots = max(test_data_means)
gamma_list = np.linspace(0,max_gamma_plots,100)

for data_index_guarantee in range(num_data_instances):
    ellipsoid = gror.Ellipsoid(dataset = train_sets[data_index_guarantee], confidence_level = 0.99)
    
    minmax_values.append([])
    
            
    for opt_instance in opt_instances:
        vals = []
        for g in gamma_list:
            ellipsoid.gamma = g*Gamma_max_value
            x_dummy, val = gror.minMaxEllipsoid(ellipsoid = ellipsoid, opt_inst = opt_instance, uncert_indices = np.arange(opt_instance.n))
            vals.append(val)
            print(val)
            
        minmax_values[data_index_guarantee].append(vals[:])


guarantee_plot = []
adv_regret_plot = []
guarantee_plot.append({
"test_data": test_data_means})
adv_regret_plot.append({
"test_data": test_data_means})

boxplot_data = []


for frac in setup_RO:
    runtimes = []

    values = 0
    instance_counter = -1
    
    y_guarantees = np.zeros(len(gamma_list))
    y_adv_regret = np.zeros(len(gamma_list))
    
    
    for opt_instance in opt_instances:
        instance_counter+=1
        for i in range(num_data_instances):
            ellipsoid = gror.Ellipsoid(dataset = train_sets[i], confidence_level = 0.99)
            ellipsoid.gamma = frac*ellipsoid.gamma
            gamma0 = ellipsoid.gamma
            
            
            start = time.time() 
            x_opt, opt_val = gror.minMaxEllipsoid(ellipsoid = ellipsoid, opt_inst = opt_instance, uncert_indices = np.arange(opt_instance.n))
            #x_opt, opt_val = gror.runRobustOptimization(dataset = train_sets[i], num_clusters = num_clusters, confidence_level = conf, opt_inst = opt_instance, uncert_indices = np.arange(opt_instance.n))
            end = time.time()
            runtime = end-start
            
            avg, std, risk, maxv, minv = gror.getSolutionStatistics(x = x_opt, test_data = test_sets[i])
            runtimes.append(runtime)
            
            values = values + np.sort(np.dot(test_sets[i],x_opt))
            
            csv_file = open(filename_output +  ".csv", 'a')
            out = csv.writer(csv_file,delimiter=";", lineterminator = "\n")
            row = ["Classic RO",dataType,frac,instance_counter,i,dim,len(train_sets[i]),len(test_sets[i]),avg,std,maxv,minv,runtime]
            out.writerow(row)
            csv_file.close()
            
            # csv_file = open("solutions.csv", 'a')
            # out = csv.writer(csv_file,delimiter=";", lineterminator = "\n")
            # row = list(x_opt)
            # out.writerow(row)
            # csv_file.close()
            
            linear_term = np.dot(ellipsoid.mean,x_opt)
            vec = np.dot(ellipsoid.matrix,x_opt-ellipsoid.mean)
            quadratic_term = np.sqrt(np.dot(x_opt-ellipsoid.mean,vec))

            for l, gamma in enumerate(gamma_list):
                if gamma*Gamma_max_value<=gamma0:
                    y_guarantees[l] += opt_val
                else:
                    y_guarantees[l] += 10**7
                    
                y_adv_regret[l] += linear_term + np.sqrt(gamma*Gamma_max_value)*quadratic_term-minmax_values[i][instance_counter][l]
            
    
    values = values / (num_data_instances*num_opt_instances)
    y_guarantees = y_guarantees / (num_data_instances*num_opt_instances)
    y_adv_regret = y_adv_regret / (num_data_instances*num_opt_instances)


    guarantee_plot.append({
    "label": "RO(" + str(np.round(frac,3)) + ")",
    "gamma": gamma_list,
    "y": y_guarantees})
    
    adv_regret_plot.append({
    "label": "RO(" + str(np.round(frac,3)) + ")",
    "gamma": gamma_list,
    "y": y_adv_regret})
    
    
    boxplot_data.append({
    "label": "RO("+ str(np.round(frac,3))+ ")",
    "values": values[:],
    "runtimes": runtimes})
    
    
for conf in setup_RO_d:
    values = 0
    runtimes = []
    
    y_adv_regret = np.zeros(len(gamma_list))
    
    instance_counter = -1
    for opt_instance in opt_instances:
        instance_counter+=1
        for i in range(num_data_instances):
            start = time.time()
            x_opt, opt_val = gror.runRobustOptimizationScenarios(dataset = train_sets[i], num_clusters = num_clusters, confidence_level = conf, opt_inst = opt_instance, uncert_indices = np.arange(opt_instance.n))
            end = time.time()
            runtime = end-start
            
            avg, std, risk, maxv, minv = gror.getSolutionStatistics(x = x_opt, test_data = test_sets[i])
            
            runtimes.append(runtime)
            
            values = values + np.sort(np.dot(test_sets[i],x_opt))
            
            csv_file = open(filename_output +  ".csv", 'a')
            out = csv.writer(csv_file,delimiter=";", lineterminator = "\n")
            row = ["Classic RO Discrete",dataType,conf,instance_counter,i,dim,len(train_sets[i]),len(test_sets[i]),avg,std,maxv,minv,runtime]
            out.writerow(row)
            csv_file.close()
            
            linear_term = np.dot(ellipsoid.mean,x_opt)
            vec = np.dot(ellipsoid.matrix,x_opt-ellipsoid.mean)
            quadratic_term = np.sqrt(np.dot(x_opt-ellipsoid.mean,vec))

            for l, gamma in enumerate(gamma_list):
                y_adv_regret[l] += linear_term + np.sqrt(gamma*Gamma_max_value)*quadratic_term-minmax_values[i][instance_counter][l]
            
    
    y_adv_regret = y_adv_regret / (num_data_instances*num_opt_instances)
    
    
    adv_regret_plot.append({
    "label": "RO$_d$("+str(np.round(conf,3))+")",
    "gamma": gamma_list,
    "y": y_adv_regret})

    
    values = values / (num_data_instances*num_opt_instances)
    
    
    boxplot_data.append({
    "label": "RO$_d$(" + str(np.round(conf,3)) + ")",
    "values": values[:],
    "runtimes": runtimes})

for tau in setup_Sat:
    values = 0
    runtimes = []
    
    y_guarantees = np.zeros(len(gamma_list))
    y_adv_regret = np.zeros(len(gamma_list))
    
    instance_counter = -1
    for opt_instance in opt_instances:
        instance_counter+=1
        for i in range(num_data_instances):
            #get f_target for robust satisficing
            ellipsoid = gror.Ellipsoid(dataset = train_sets[i], confidence_level = 0.99)
            ellipsoid.gamma = 0
            
            x_nom, nom_val = gror.minMaxEllipsoid(ellipsoid = ellipsoid, opt_inst = opt_instance, uncert_indices = np.arange(opt_instance.n))
            print("Nominal Value:",nom_val)
            #x_nom, nom_val = gror.runRobustOptimization(dataset = train_sets[i], num_clusters = num_clusters, confidence_level = 0, opt_inst = opt_instance, uncert_indices = np.arange(opt_instance.n))
            
            start = time.time() 
            if nom_val >=0:
                target = tau*nom_val
            else:
                target = (1.0 - (tau-1.0))*nom_val
                
            x_opt, opt_val = gror.runRobustSatisficing(dataset = train_sets[i], f_target = target, opt_inst = opt_instance, uncert_indices = np.arange(opt_instance.n))
            end = time.time()
            runtime = end-start
            
            avg, std, risk, maxv, minv = gror.getSolutionStatistics(x = x_opt, test_data = test_sets[i])
            
            runtimes.append(runtime)
            
            values = values + np.sort(np.dot(test_sets[i],x_opt))
            
            csv_file = open(filename_output +  ".csv", 'a')
            out = csv.writer(csv_file,delimiter=";", lineterminator = "\n")
            row = ["RO_Sat",dataType,tau,instance_counter,i,dim,len(train_sets[i]),len(test_sets[i]),avg,std,maxv,minv,runtime]
            out.writerow(row)
            csv_file.close()
            
            linear_term = np.dot(ellipsoid.mean,x_opt)
            vec = np.dot(ellipsoid.matrix,x_opt-ellipsoid.mean)
            quadratic_term = np.sqrt(np.dot(x_opt-ellipsoid.mean,vec))
                
            for l, gamma in enumerate(gamma_list):
                y_guarantees[l] += opt_val*(gamma*Gamma_max_value) + tau*nom_val
                y_adv_regret[l] += linear_term + np.sqrt(gamma*Gamma_max_value)*quadratic_term-minmax_values[i][instance_counter][l]
            
    
    
    y_guarantees = y_guarantees / (num_data_instances*num_opt_instances)  
    y_adv_regret = y_adv_regret / (num_data_instances*num_opt_instances)

    
    guarantee_plot.append({
    "label": "SAT("+str(tau)+")",
    "gamma": gamma_list,
    "y": y_guarantees})
    
    adv_regret_plot.append({
    "label": "SAT("+str(tau)+")",
    "gamma": gamma_list,
    "y": y_adv_regret})
    
    values = values / (num_data_instances*num_opt_instances)

    boxplot_data.append({
    "label": "Sat("+str(tau)+")",
    "values": values[:],
    "runtimes": runtimes})
    
for frac in setup_REG:
    values = 0
    runtimes = []
    
    y_adv_regret = np.zeros(len(gamma_list))
    
    instance_counter = -1
    for opt_instance in opt_instances:
        instance_counter+=1
        for i in range(num_data_instances):
            ellipsoid = gror.Ellipsoid(dataset = train_sets[i], confidence_level = 0.99)
            ellipsoid.gamma = frac*ellipsoid.gamma
            
            start = time.time()
            x_opt, opt_val = gror.runClassicalRegret(dataset = train_sets[i], num_clusters = num_clusters, gamma_val = ellipsoid.gamma, opt_inst = opt_instance, uncert_indices = np.arange(opt_instance.n))
            end = time.time()
            runtime = end-start
            
            avg, std, risk, maxv, minv = gror.getSolutionStatistics(x = x_opt, test_data = test_sets[i])
            
            runtimes.append(runtime)
            
            values = values + np.sort(np.dot(test_sets[i],x_opt))
            
            csv_file = open(filename_output +  ".csv", 'a')
            out = csv.writer(csv_file,delimiter=";", lineterminator = "\n")
            row = ["Classic Regret",dataType,frac,instance_counter,i,dim,len(train_sets[i]),len(test_sets[i]),avg,std,maxv,minv,runtime]
            out.writerow(row)
            csv_file.close()
            
            linear_term = np.dot(ellipsoid.mean,x_opt)
            vec = np.dot(ellipsoid.matrix,x_opt-ellipsoid.mean)
            quadratic_term = np.sqrt(np.dot(x_opt-ellipsoid.mean,vec))
                
            for l, gamma in enumerate(gamma_list):
                y_adv_regret[l] += linear_term + np.sqrt(gamma*Gamma_max_value)*quadratic_term-minmax_values[i][instance_counter][l]
            

    
    values = values / (num_data_instances*num_opt_instances)
    y_adv_regret = y_adv_regret / (num_data_instances*num_opt_instances)
    
    adv_regret_plot.append({
    "label": "Reg("+str(np.round(frac,3))+")",
    "gamma": gamma_list,
    "y": y_adv_regret})
    
    
    boxplot_data.append({
    "label": "Reg(" + str(np.round(frac,3)) + ")",
    "values": values[:],
    "runtimes": runtimes}) 


for p in setup_GARO:
    values = 0
    runtimes = []
    
    y_guarantees = np.zeros(len(gamma_list))
    y_adv_regret = np.zeros(len(gamma_list))
    
    instance_counter = -1
    for opt_instance in opt_instances:
        instance_counter+=1
        for i in range(num_data_instances):
            start = time.time() 
            x_opt, opt_val = gror.runGARO(dataset = train_sets[i], p=p, opt_inst = opt_instance, uncert_indices = np.arange(opt_instance.n))
            end = time.time()
            runtime = end-start
            
            avg, std, risk, maxv, minv = gror.getSolutionStatistics(x = x_opt, test_data = test_sets[i])
            
            runtimes.append(runtime)
            
            values = values + np.sort(np.dot(test_sets[i],x_opt))
            
            csv_file = open(filename_output +  ".csv", 'a')
            out = csv.writer(csv_file,delimiter=";", lineterminator = "\n")
            row = ["GARO",dataType,p,instance_counter,i,dim,len(train_sets[i]),len(test_sets[i]),avg,std,maxv,minv,runtime]
            out.writerow(row)
            csv_file.close()
            
            linear_term = np.dot(ellipsoid.mean,x_opt)
            vec = np.dot(ellipsoid.matrix,x_opt-ellipsoid.mean)
            quadratic_term = np.sqrt(np.dot(x_opt-ellipsoid.mean,vec))

            for l, gamma in enumerate(gamma_list):
                y_guarantees[l] += opt_val*((1+gamma*Gamma_max_value)**p) + minmax_values[i][instance_counter][l]
                y_adv_regret[l] += linear_term + np.sqrt(gamma*Gamma_max_value)*quadratic_term-minmax_values[i][instance_counter][l]
    
    
    y_guarantees = y_guarantees / (num_data_instances*num_opt_instances)
    
    y_adv_regret = y_adv_regret / (num_data_instances*num_opt_instances)
    
    guarantee_plot.append({
    "label": "GARO("+str(p)+")",
    "gamma": gamma_list,
    "y": y_guarantees})
    
    adv_regret_plot.append({
    "label": "GARO("+str(p)+")",
    "gamma": gamma_list,
    "y": y_adv_regret})
    
    values = values / (num_data_instances*num_opt_instances)
    
    
    boxplot_data.append({
    "label": "GARO("+str(p)+")",
    "values": values[:],
    "runtimes": runtimes})
    


with open(output_folder + "\\guarantee_data.pkl", "wb") as f:
    pickle.dump(guarantee_plot, f)
    
with open(output_folder + "\\adv_regret_data.pkl", "wb") as f:
    pickle.dump(adv_regret_plot, f)
    
with open(output_folder + "\\boxplot_data.pkl", "wb") as f:
    pickle.dump(boxplot_data, f)


