import numpy as np
import numpy.ma as ma
import time
import pickle
import os
import copy
import GFA 
import argparse
from utils import GFAtools
from visualization import results_simulations

def get_data(args, infoMiss=False):
    # Generate some data from the generative model, with pre-specified
    # latent components
    Ntrain = 400; Ntest = 100
    N = Ntrain + Ntest #number of samples
    M = args.num_sources  #number of groups/data sources
    d = np.array([50, 30]) #number of dimensios for each group
    true_K = 4  # true latent components
    #Manually specify Z 
    Z = np.zeros((N, true_K))
    for i in range(0, N):
        Z[i,0] = np.sin((i+1)/(N/20))
        Z[i,1] = np.cos((i+1)/(N/20))
        Z[i,2] = 2 * ((i+1)/N-0.5)    
    Z[:,3] = np.random.normal(0, 1, N)          
    #Noise precisions
    tau = [[] for _ in range(d.size)]
    tau[0] = 5 * np.ones((1,d[0]))[0] 
    tau[1] = 10 * np.ones((1,d[1]))[0]
    #ARD parameters
    alpha = np.zeros((M, true_K))
    alpha[0,:] = np.array([1,1,1e6,1])
    alpha[1,:] = np.array([1,1,1,1e6])     
    #W and X
    W = [[] for _ in range(d.size)]
    X_train = [[] for _ in range(d.size)]
    X_test = [[] for _ in range(d.size)]
    for i in range(0, d.size):
        W[i] = np.zeros((d[i], true_K))
        for t in range(0, true_K):
            #generate W from p(W|alpha)
            W[i][:,t] = np.random.normal(0, 1/np.sqrt(alpha[i,t]), d[i])
        X = np.zeros((N, d[i]))
        for j in range(0, d[i]):
            #generate X from the generative model
            X[:,j] = np.dot(Z,W[i][j,:].T) + \
            np.random.normal(0, 1/np.sqrt(tau[i][j]), N*1)    
        #Get train and test data
        X_train[i] = X[0:Ntrain,:] #Train data
        X_test[i] = X[Ntrain:N,:] #Test data
    #Latent variables for training the model    
    Z = Z[0:Ntrain,:]

    #Generate incomplete training data
    if args.scenario == 'incomplete':
        missing_Xtrue = [[] for _ in range(len(infoMiss['group']))]
        for i in range(len(infoMiss['group'])): 
            if 'random' in infoMiss['type'][i]: 
                #remove entries randomly
                missing_val =  np.random.choice([0, 1], 
                            size=(X_train[infoMiss['group'][i]-1].shape[0],d[infoMiss['group'][i]-1]), 
                            p=[1-infoMiss['perc'][i-1]/100, infoMiss['perc'][i-1]/100])
                mask_miss =  ma.array(X_train[infoMiss['group'][i]-1], mask = missing_val).mask
                missing_Xtrue[i] = np.where(missing_val==1, X_train[infoMiss['group'][i]-1],0)
                X_train[infoMiss['group'][i]-1][mask_miss] = 'NaN'
            elif 'rows' in infoMiss['type'][i]: 
                #remove rows randomly
                missing_Xtrue[i] = np.zeros((Ntrain,d[i]))
                n_rows = int(infoMiss['perc'][i-1]/100 * X_train[infoMiss['group'][i]-1].shape[0])
                shuf_samples = np.arange(Ntrain)
                np.random.shuffle(shuf_samples)
                missing_Xtrue[i][shuf_samples[0:n_rows],:] = X_train[infoMiss['group'][i]-1][shuf_samples[0:n_rows],:]
                X_train[infoMiss['group'][i]-1][shuf_samples[0:n_rows],:] = 'NaN'
    #Store data            
    data = {'X_tr': X_train, 'X_te': X_test, 'W': W, 'Z': Z, 'tau': tau, 'alpha': alpha, 'true_K': true_K}
    if args.scenario == 'incomplete':
        data.update({'trueX_miss': missing_Xtrue}) 
    return data        

def main(args):
    #info to generate incomplete data sets
    if args.scenario == 'incomplete':
        infmiss = {'perc': [10, 20], #percentage of missing data 
                'type': ['rows', 'random'], #type of missing data 
                'group': [1, 2]} #groups that will have missing values            

    #Make directory to save the results of the experiments          
    res_dir = f'results/2groups/GFA_{args.noise}/{args.K}comps/{args.scenario}'
    if not os.path.exists(res_dir):
            os.makedirs(res_dir)
    for init in range(0, args.num_runs):
        print("Run:", init+1)
        #-GENERATE DATA
        #------------------------------------------------------
        data_file = f'{res_dir}/[{init+1}]Data.dictionary'
        if not os.path.exists(data_file):
            print("Generating data---------")
            if args.scenario == 'complete':
                simData = get_data(args)
            else:
                simData = get_data(args, infmiss)
            #Save file with generated data
            with open(data_file, 'wb') as parameters:
                    pickle.dump(simData, parameters)
            print("Data generated!")            
        else:
            with open(data_file, 'rb') as parameters:
                simData = pickle.load(parameters)
            print("Data loaded!")         

        #-RUN MODEL
        #---------------------------------------------------------------------------------         
        res_file = f'{res_dir}/[{init+1}]ModelOutput.dictionary'
        if not os.path.exists(res_file):  
            print("Running the model---------")
            X_tr = simData['X_tr']
            if 'diagonal' in args.noise:    
                GFAmodel = GFA.IncompleteDataModel(X_tr, args.K)
            else:
                assert args.scenario == 'complete'
                GFAmodel = GFA.OriginalModel(X_tr, args.K)      
            #Fit model
            time_start = time.process_time()
            GFAmodel.fit(X_tr)
            GFAmodel.time_elapsed = time.process_time() - time_start
            print(f'Computational time: {float("{:.2f}".format(GFAmodel.time_elapsed))}s')
            
            #-Predictions (Predict group 2 from group 1) 
            #------------------------------------------------------------------------------
            #Compute mean squared error
            obs_group = np.array([1, 0]) #group 1 was observed 
            gpred = np.where(obs_group == 0)[0][0] #get the non-observed groups
            X_test = simData['X_te'][gpred]
            X_pred = GFAtools(simData['X_te'], GFAmodel).PredictView(obs_group, args.noise)
            GFAmodel.MSE = np.mean((X_test - X_pred[0]) ** 2)
            #MSE - chance level (MSE between test values and train means)
            Tr_means = np.ones((X_test.shape[0], X_test.shape[1])) * \
                np.nanmean(simData['X_tr'][gpred], axis=0)           
            GFAmodel.MSE_chlev = np.mean((X_test - Tr_means) ** 2)
            
            #Predict missing values
            if args.scenario == 'incomplete':
                Corr_miss = np.zeros((1,len(infmiss['group'])))
                missing_pred = GFAtools(simData['X_tr'], GFAmodel).PredictMissing(args.num_sources, infmiss)
                missing_true = simData['trueX_miss']
                for i in range(len(infmiss['group'])):
                    Corr_miss[0,i] = np.corrcoef(missing_true[i][missing_true[i] != 0], 
                                        missing_pred[i][missing_pred[i] != 0])[0,1]                   
                GFAmodel.Corr_miss = Corr_miss

            #Save file containing results
            with open(res_file, 'wb') as parameters:
                pickle.dump(GFAmodel, parameters)
        else:
            #Load file containing results
            with open(res_file, 'rb') as parameters:
                GFAmodel = pickle.load(parameters) 

        #Impute median before training the model
        if args.impMedian: 
            X_impmed = copy.deepcopy(X_train) 
            for i in range(len(remove)):
                if vmiss[i] == 1:
                    miss_view = np.array([0, 1])
                elif vmiss[i] == 2:
                    miss_view = np.array([1, 0])
                vpred = np.array(np.where(miss_view == 0))
                for j in range(X_median[i].size):
                    X_impmed[vpred[0,0]][np.isnan(X_impmed[vpred[0,0]][:,j]),j] = np.nanmedian(X_train[infmiss['group'][i]-1],axis=0)
            
            print("Run Model after imp. median----------")
            GFAmodel2[init] = GFA.MissingModel(X_impmed, k)
            L = GFAmodel2[init].fit(X_impmed)
            GFAmodel2[init].L = L
            GFAmodel2[init].k_true = T
            X_pred = [[] for _ in range(d.size)]
            X_pred[vpred2[0,0]] = GFAtools(X_test, GFAmodel2[init], obs_view2).PredictView('diagonal')

            #-Metrics
            #MSE - predict view 2 from view 1 
            MSE2 = np.mean((X_test[vpred2[0,0]] - X_pred[vpred2[0,0]]) ** 2)    
            GFAmodel2[init].MSE2 = MSE2
   
            #Save file
            median_path = f'{res_dir}/[{init+1}]ModelOutput_median.dictionary'
            with open(median_path, 'wb') as parameters:
                pickle.dump(GFAmodel2, parameters)          

    #visualization
    #results_simulations(args.num_runs, res_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GFA with two groups")
    parser.add_argument("--scenario", nargs='?', default='incomplete', type=str)
    parser.add_argument("--noise", nargs='?', default='diagonal', type=str)
    parser.add_argument("--num-sources", nargs='?', default=2, type=int)
    parser.add_argument("--K", nargs='?', default=6, type=int)
    parser.add_argument("--num-runs", nargs='?', default=2, type=int)
    parser.add_argument("--impMedian", nargs='?', default=False, type=bool)
    args = parser.parse_args()

    main(args)    


