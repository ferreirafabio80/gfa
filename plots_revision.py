import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

path = 'results/revision_plots/'

# Create bar plots with MSEs
#---------------------------------------------------
# Predict X1 from X2
#ours
MSEour_means = [1.38, 1.23, 1.14]
MSEour_stds = [0.21, 0.25, 0.19]
#chance level
MSEch_means = [2.48, 2.29, 2.27]
MSEch_stds = [0.28, 0.27, 0.26]
#imputation
MSEimp_means = [1.27, 1.17]
MSEimp_stds = [0.25, 0.18]

fig, ax = plt.subplots()
x_c = np.arange(3)
width = 0.25
ax.bar(x_c, MSEch_means, yerr=MSEch_stds, width = width, color='b', alpha=0.5, ecolor='black', capsize=10)
ax.bar(x_c+width, MSEour_means, yerr=MSEour_stds, width=width, color='r', alpha=0.5, ecolor='black', capsize=10)
ax.bar(np.arange(2)+1+2*width, MSEimp_means, yerr=MSEimp_stds, width = width, color='g', alpha=0.5, ecolor='black', capsize=10)
ax.set_xticks(x_c+width)
ax.set_ylim([0,3.5])
ax.set_xticklabels(['Experiment 1', 'Experiment 2a', 'Experiment 2b'])
plt.legend(['chance','ours','imputation'])
plt.savefig(f'{path}barplot_x1fromx2.svg'); plt.close()

# Predict X2 from X1
#ours
MSEour_means = [0.81, 0.71, 0.75]
MSEour_stds = [0.18, 0.11, 0.18]
#chance level
MSEch_means = [2.24, 2.06, 2.22]
MSEch_stds = [0.39, 0.29, 0.36]
#imputation
MSEimp_means = [0.74, 0.75]
MSEimp_stds = [0.11, 0.18]

fig, ax = plt.subplots()
x_c = np.arange(3)
width = 0.25
ax.bar(x_c, MSEch_means, yerr=MSEch_stds, width = width, color='b', alpha=0.5, ecolor='black', capsize=10)
ax.bar(x_c+width, MSEour_means, yerr=MSEour_stds, width=width, color='r', alpha=0.5, ecolor='black', capsize=10)
ax.bar(np.arange(2)+1+2*width, MSEimp_means, yerr=MSEimp_stds, width = width, color='g', alpha=0.5, ecolor='black', capsize=10)
ax.set_xticks(x_c+width)
ax.set_ylim([0,3.5])
ax.set_xticklabels(['Experiment 1', 'Experiment 2a', 'Experiment 2b'])
plt.legend(['chance','ours','imputation'])
plt.savefig(f'{path}barplot_x2fromx1.svg'); plt.close()

# Create plot of performance of different percentage of missingness
#------------------------------------------------------------------
# Random missingness
mean_nomissv1 = np.array([1.27, 1.21, 1.18, 1.26, 1.43])
std_nomissv1 = np.array([0.23, 0.26, 0.17, 0.28, 0.37])
mean_miss20v1 = np.array([1.21, 1.19, 1.08, 1.40, 1.70])
std_miss20v1 = np.array([0.22, 0.21, 0.21, 0.48, 0.39])

x = np.arange(5)
plt.errorbar(x, mean_nomissv1, yerr = std_nomissv1)
plt.errorbar(x, mean_miss20v1, yerr = std_miss20v1)
plt.legend(['No missing data in group 1','20% missing rows in group 1'])
plt.xticks([0,1,2,3,4],['0%', '20%', '40%', '60%','80%']); plt.ylim(0, 3)
plt.xlabel('Percentage of missing data in group 2') 
plt.ylabel('MSE')
plt.savefig(f'{path}lineplot_random.svg'); plt.close()

# Nonrandom missingness
mean_nomissv1 = np.array([1.27, 1.21, 1.35, 2.20, 5.93])
std_nomissv1 = np.array([0.23, 0.24, 0.19, 0.33, 1.29])
mean_miss20v1 = np.array([1.21, 1.26, 1.44, 2.50, 7.41])
std_miss20v1 = np.array([0.22, 0.28, 0.25, 0.58, 1.77])

plt.errorbar(x, mean_nomissv1, yerr = std_nomissv1)
plt.errorbar(x, mean_miss20v1, yerr = std_miss20v1)
plt.legend(['No missing data in group 1', '20% missing rows in group 1'])
plt.xticks([0,1,2,3,4],['0%', '13%', '32%', '44%','64%']); plt.ylim(0, 12)
plt.xlabel('Percentage of missing data in group 2') 
plt.ylabel('MSE')
plt.savefig(f'{path}lineplot_nonrandom.svg'); plt.close()

# Create scree plot of the variance explained by each factor
#------------------------------------------------------------------
data = pd.read_excel(f'{path}Info_factors_complete.xlsx')
var_total = data['Var_total'].to_numpy()
ratio = data['Ratio'].to_numpy()

plt.plot(np.arange(ratio.size), var_total, alpha=0.7)
plt.xlabel('Factors', fontsize=8) 
plt.ylabel('Percentage of variance explained', fontsize=8); plt.ylim(0, 1.1)
plt.yticks(fontsize=5)
plt.xticks([0, 1, 14, 27, 42, 43],['BS a', 'BS b', 'Sh a', 'Sh c', 'Sh d', 'Sh b'], 
        rotation= 90, fontsize=5)
plt.vlines(x = [0, 1, 14, 27, 42, 43], ymin=[0, 0, 0, 0, 0, 0], 
        ymax=[1.1, 1.1, 1.1, 1.1, 1.1, 1.1], alpha=0.4, colors='red')
plt.savefig(f'{path}lineplot_variance.svg'); plt.close()

plt.scatter(np.arange(ratio.size), ratio, color='green', alpha=0.7)
plt.xlabel('Factors', fontsize=10) 
plt.ylabel('Ratio', fontsize=10)
plt.xticks([0, 1, 14, 27, 42, 43],['BS a', 'BS b', 'Sh a', 'Sh c', 'Sh d', 'Sh b'], 
        rotation= 90, fontsize=5)
plt.axhline(y=0.001, color='r')        
plt.yticks(fontsize=5)
plt.yscale('log')
plt.savefig(f'{path}scatterplot_rat.svg'); plt.close()

res_dir = '/Users/fabioferreira/Desktop/repos/gfa/results/HCP/1000subjs/experiments/GFA_diagonal/80models_old/complete/training80'
n_comps = 75
L = []
for j in range(n_comps):
        red_file = f'{res_dir}/Reduced_model_updated_{j+1}comps.dictionary'
        
        with open(red_file, 'rb') as parameters:
                Redmodel = pickle.load(parameters)

        L.append(Redmodel.L[-1])
        #print(Redmodel.k)

plt.plot(np.arange(1,n_comps+1,1), L, color='black', alpha = 0.7)
plt.xlabel('Number of factors', fontsize=10) 
plt.ylabel('ELBO', fontsize=10)
plt.yticks(fontsize=5)
plt.xticks(fontsize=5) #[6, 75],['6', '75'], rotation= 90,      
plt.savefig(f'{path}ELBOs.svg'); plt.close()        




