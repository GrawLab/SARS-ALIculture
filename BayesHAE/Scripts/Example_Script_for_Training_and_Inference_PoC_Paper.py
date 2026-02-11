# ============================================================================
# BAYESFLOW Training Script Example
# ============================================================================
# This script trains a BayesFlow neural network for amortized Bayesian inference 
# on pre-computed Morpheus simulation data to infer infection kinetics.
#
# Disclaimer: This is an example script to showcase the general workflow of training BayesFlow neural posterior estimator for 
# the HOM, HAE and HAE-Phi models. Expect for the training and validation data loading parts and the simulator the rest of the code is largely model-agnostic.
# Users should adapt the data loading and simulator functions to their own models and data.
# ============================================================================
# Author: Pascal Lukas
# ============================================================================
# IMPORTS
# ============================================================================

# Standard library imports
import datetime
from functools import partial
import os
import os.path
import pickle
import logging
import argparse
from random import sample
import numpy as np
import pandas as pd

# BayesFlow framework 
import bayesflow.diagnostics as diag
from bayesflow.amortizers import AmortizedPosterior
from bayesflow.networks import InvertibleNetwork, SequenceNetwork, DeepSet, TimeSeriesTransformer
from bayesflow.simulation import GenerativeModel, Prior, Simulator
from bayesflow.trainers import Trainer

# Image processing and analysis
from skimage.io import imread
from skimage.measure import label 
import pyclesperanto_prototype as cle 
import networkx as nx 

# Deep learning frameworks
import tensorflow as tf
import tensorflow_probability as tfp

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# Statistical tools
from scipy.stats import binom, median_abs_deviation
from sklearn.metrics import confusion_matrix, r2_score

# Configure logging
logging.basicConfig()

# Random number generator
#RNG = np.random.default_rng(2023)


# ============================================================================
# COMMAND LINE ARGUMENT PARSING
# ============================================================================

# Set up argument parser for hyperparameter configuration
parser = argparse.ArgumentParser(description='Provide Hyperparameters')

parser.add_argument('SD', metavar='SD',
                    help=' Summary Dimensions')
parser.add_argument('CL', metavar='CL',
                    help=' Coupling Layer')
parser.add_argument('DO', metavar='DO',
                    help=' Dropout Rate')
parser.add_argument('LR', metavar='LR',
                    help=' Learning Rate')
parser.add_argument('Epochs', metavar='Epochs',
                    help='Number of training epochs')

# Parse arguments
args = parser.parse_args()

# Assign hyperparameters from command line arguments
SD = args.SD
CL = args.CL
DO = args.DO
LR = args.LR
Epochs = args.Epochs



# ============================================================================
# LOAD PREPROCESSED DATA
# ============================================================================

# Load the simulated observational data array for testing
with open("obs_data_array_XXX.pkl", 'rb') as f:
    obs_data_array = pickle.load(f)

# Load the simulated testing parameter array
with open("testing_param_array_XXX.pkl", 'rb') as f:
    testing_param_array = pickle.load(f)


# Load dictionary containing both simulated observational data and parameter arrays
with open("obs_data_array_dict_XXX.pkl", 'rb') as f:
    obs_data_array_dict = pickle.load(f)


# ============================================================================
# PARAMETER COMBINATION FUNCTIONS
# ============================================================================

def get_PC_given_sampled_KDE(params):
    """
    Find the closest parameter combination (PC) number from a precomputed list. This allows for matching a sampled parameter
    combination with its respected pre-run simulation data.
    This current script only accounts for 3 parameters: Rho_adj, B_Adj, CC_CF_Ratio. Adapt as needed for more parameters.
    
    """
    # Unpack parameters
    rho_adj, b_adj, cc_cf_ratio = params
    
    # Load the parameter combination lookup table (10k combinations)
    combination_list = pd.read_csv("Parameter_Combination_table.tsv", sep="\t")
    
    # Step 1: Filter to 100 closest matches by Rho_adj
    dist_filt1 = combination_list.iloc[(combination_list['Rho_adj'] - rho_adj).abs().argsort()[:100]]
    
    # Step 2: From those, find 10 closest by B_Adj
    dist_filt2 = dist_filt1.iloc[(dist_filt1['B_Adj'] - b_adj).abs().argsort()[:10]]
    
    # Step 3: From those, find the single closest match by CC_CF_Ratio
    PC_row = dist_filt2.iloc[(dist_filt2['CC_CF_Ratio'] - cc_cf_ratio).abs().argsort()[:1]]

    
    # Extract the PC number from the first column of the matched row
    PC_number = PC_row.iloc[0, 0]

    return PC_number

def model_prior(batch_size):
    """
    Generates a random draw from the prior distribution. In this example only Rho, Beta and CC_CF_Ratio are varied.
    
    """
    # Load parameter combination lookup table
    combination_list = pd.read_csv("Parameter_Combination_table.tsv", sep="\t")
    
    # Randomly sample PC indices from the valid set 
    sample_pc = sample(list(range(1, len(combination_list) + 1)), batch_size)
    
    # Repeat each sample once (placeholder for potential future expansion)
    sample_pc = np.repeat(sample_pc, 1)
    
    # Retrieve parameter values for sampled PCs (subtract 1 for 0-based indexing)
    sample_params = combination_list.loc[np.array(sample_pc) - 1]
    
    # Extract individual parameter columns
    rho_adj = sample_params["Rho_adj"]
    b_adj = sample_params["B_Adj"]
    cc_cf_ratio = sample_params["CC_CF_Ratio"]

    # Stack parameters into a batch array and reshape to (batch_size, 3)
    param_batch = np.array([rho_adj, b_adj, cc_cf_ratio]).T.reshape(batch_size * 1, 3)

    return param_batch


# ============================================================================
# FUNCTIONS FOR MORPHEUS SIMULATION OUTPUTS
# ============================================================================

def retrieve_morpheus_logger_from_model_list_with_params_for_BF(PC, Rep, param_dir):
    """
    Retrieve Morpheus simulation logger data for BayesFlow analysis. The PC number is matched based on the sampled prior.

    """
    # Construct filename for logger data
    folder_logger = param_dir + "Logger_2_PC_S" + str(PC) + "_R" + str(Rep) + ".csv"
    
    # Read and return logger data
    logger_data = pd.DataFrame(pd.read_csv(folder_logger, sep=","))

    return logger_data


def retrieve_morpheus_cluster_from_model_list_with_params_for_BF(PC, Rep, param_dir):
    """
    Retrieve Morpheus simulation cluster analysis data for BayesFlow. The PC number is matched based on the sampled prior.
    
    """
    # Construct filename for cluster data
    folder_logger = param_dir + "Cluster_table_PC_S" + str(PC) + "_R" + str(Rep) + ".csv"
    
    # Read and return cluster data
    logger_data = pd.DataFrame(pd.read_csv(folder_logger, sep=","))

    return logger_data


def retrieve_morpheus_TEER_from_model_list_with_params_for_BF(PC, Rep, param_dir):
    """
    Retrieve Morpheus simulation TEER (Trans-Epithelial Electrical Resistance) data. The PC number is matched based on the sampled prior.
    
    """
    # Construct filename for TEER data
    folder_logger = param_dir + "TEER_table_PC_S" + str(PC) + "_R" + str(Rep) + ".csv"
    
    # Read and return TEER data
    logger_data = pd.DataFrame(pd.read_csv(folder_logger, sep=","))

    return logger_data


# ============================================================================
# SPATIAL CLUSTERING ANALYSIS FUNCTIONS
# ============================================================================

def create_graphs(filtered_infected_voronoi):
    """
    Create a graph representation of spatial cell connectivity.
    
    """
    # Generate touch matrix 
    mat = cle.generate_touch_matrix(filtered_infected_voronoi)
    
    # Convert to networkx graph 
    G = nx.from_numpy_array(np.array(mat[1:, 1:]))

    # Find all connected components (clusters of touching cells)
    conn_comp = list(nx.connected_components(G))

    return conn_comp


def calc_filter_clusters(conn_comp, cluster_threshold=3):
    """
    Filter and analyze clusters based on size threshold.
    
    """
    # Calculate size of each connected component
    length_of_conn_comp = []
    for conn in conn_comp:
        length_of_conn_comp.append(len(conn))

    # Find indices of clusters meeting the size threshold
    indexes_cluster = np.where(np.array(length_of_conn_comp) >= cluster_threshold)[0]
    
    # Extract sizes of significant clusters
    cluster_size = np.array(length_of_conn_comp)[indexes_cluster]

    return indexes_cluster, cluster_size



# ============================================================================
# PRIOR DISTRIBUTION SETUP
# ============================================================================

# Create prior distribution BF object
prior = Prior(batch_prior_fun=model_prior, 
            param_names=[r"$\rho_{Adj}$",      # Adjusted viral production rate
                        r"$B_{Adj}$",          # Adjusted infectivity rate
                        r"$Ratio_{CC-CF}$"])  # Weighting factor for cell-to-cell vs. cell-free infection

# Estimate mean and standard deviation of prior for normalization
prior_means, prior_stds = prior.estimate_means_and_stds()


# ============================================================================
# SIMULATOR FUNCTION
# ============================================================================

def simulator_morpheus(params):
    """
    Main simulator function that retrieves precomputed Morpheus simulation data.
    
    This function is based on a disrupted sampling approach, where we assume that at each timepoint and replicate a new independet tissue
    is used experimentally. Thus, we sample different replicates at each timepoint to reflect this variability. Each replicate originates from a 
    new simulation run with the same parameter combination but different random seed.
    
    The function samples 5 replicates per parameter combination and collects
    data at 5 timepoints: 96, 192, 288, 384, 480 simulation time units.
    
    Omit or delete the Cluster and TEER information when using non-spatial data.

    """
    # Initialize combined dataframe for all parameter combinations
    combined_full_logger_cluster_sample = pd.DataFrame()
    
    # Process each parameter combination
    for param_comb in params:
        # Find the closest parameter combination in the lookup table
        PC_sample = get_PC_given_sampled_KDE(param_comb)
    
        # Initialize dataframes for different data types
        logger_sample = pd.DataFrame()     
        cluster_sample = pd.DataFrame()    
        teer_sample = pd.DataFrame()       

        # Initialize temporary slices for each timepoint
        logger_sample_slice = pd.DataFrame()
        cluster_sample_slice = pd.DataFrame()
        teer_sample_slice = pd.DataFrame()

        # Loop through 5 timepoints 
        # Times: 96, 192, 288, 384, 480 correspond to 18, 42, 66, 90, 114 hours post infection
        for i in [96, 192, 288, 384, 480]:
            # Randomly select 5 replicates from available 8 runs
            rep_list = sample(range(1, 9), 5)
         
            print(rep_list)
            
            # ======== Process Logger Data for First Replicate ========
            logger_data = retrieve_morpheus_logger_from_model_list_with_params_for_BF(
                PC_sample, rep_list[0], "../Logger_Files/")
            
            # Calculate total cell count across all cell types
            total_count = list()
            for index, row in logger_data.iterrows():
                
                total_count.append(np.nansum(row[2:5]))

            # Add computed columns to logger data
            logger_data["Total_cell_count"] = total_count
            
            # Calculate fraction of infected cells (infected + infectious)
            logger_data["frac_infected"] = (
                logger_data["celltype.ciliated_infected.size"] + 
                logger_data["celltype.ciliated_infectious.size"]
            ) / logger_data["Total_cell_count"]

            # Filter to current timepoint and select relevant columns
            logger_Data_filt = logger_data[logger_data["time"] == i]
            logger_Data_filt = logger_Data_filt[[
                "time", "frac_infected", "total_viral_load", 
                "inf_neighbors_x", "inf_neighbors_y" #inf_neighbors_x = Infected Neighbors; inf_neighbors_y = Infectious Neighbors SD
            ]]
            logger_sample_slice = logger_Data_filt

            # ======== Process Cluster Data for First Replicate ========
            cluster_data = retrieve_morpheus_cluster_from_model_list_with_params_for_BF (PC_sample,rep_list [0], "../Cluster_Analysis/")
            
            # Filter to current timepoint and select cluster statistics
            cluster_data_filt = cluster_data [cluster_data ["time"] == i]
            cluster_data_filt = cluster_data_filt [["time", "Mean_Cluster_Size","SD_Cluster_Size", "Mean_Cluster_Count", "Mean_Cluster_Distance", "SD_Cluster_Distance"]]
            cluster_sample_slice = cluster_data_filt

            # ======== Process TEER Data for First Replicate ========
            teer_data = retrieve_morpheus_TEER_from_model_list_with_params_for_BF(PC_sample,rep_list [0], "../TEER/")
            
            # Remove duplicate timepoints (keep first occurrence)
            TEER_df_filt = teer_data.groupby('Timepoint').first().reset_index()  
            
            # Filter to current timepoint
            TEER_data_filt = TEER_df_filt [TEER_df_filt ["Timepoint"] == i]
            
            # Normalize TEER by baseline value (6.05). Baseline from simulated uninfected HAE tissue.
            # Relative TEER < 1 indicates barrier damage
            TEER_data_filt.loc[:,"Rel. TEER"] = TEER_data_filt ["TEER"] /6.05
            TEER_data_filt = TEER_data_filt [["Timepoint", "Rel. TEER"]]

            teer_sample_slice = TEER_data_filt


            # ======== Process Remaining Replicates (2-5) ========
            # Concatenate data from additional replicates horizontally
            for rep in rep_list [1::] :

                # Logger data for additional replicate
                logger_data = retrieve_morpheus_logger_from_model_list_with_params_for_BF (PC_sample,rep, "../Logger_Files/")

                # Calculate total cell count
                total_count = list()
                for index, row in logger_data.iterrows():

                    total_count.append(np.nansum(row [2:5]))

                # Add computed columns
                logger_data ["Total_cell_count"] = total_count
                logger_data ["frac_infected"] = (logger_data["celltype.ciliated_infected.size"] + logger_data["celltype.ciliated_infectious.size"])/ logger_data ["Total_cell_count"]

                # Filter and drop time column 
                logger_Data_filt = logger_data [logger_data ["time"] == i]
                logger_Data_filt = logger_Data_filt.drop(["time"], axis=1)
                logger_Data_filt = logger_Data_filt [["frac_infected", "total_viral_load", "inf_neighbors_x", "inf_neighbors_y"]]
                
                # Concatenate
                logger_sample_slice =  pd.concat([logger_sample_slice, logger_Data_filt], axis=1)

                # Cluster data 
                cluster_data = retrieve_morpheus_cluster_from_model_list_with_params_for_BF (PC_sample,rep, "../Cluster_Analysis/")
                
                # Filter and drop time column
                cluster_data_filt = cluster_data [cluster_data ["time"] == i]
                cluster_data_filt = cluster_data_filt.drop(["time"], axis=1)
                cluster_data_filt = cluster_data_filt [["Mean_Cluster_Size","SD_Cluster_Size", "Mean_Cluster_Count", "Mean_Cluster_Distance", "SD_Cluster_Distance"]]
                
                # Concatenate
                cluster_sample_slice =  pd.concat([cluster_sample_slice, cluster_data_filt], axis=1)

                # TEER data
                teer_data = retrieve_morpheus_TEER_from_model_list_with_params_for_BF(PC_sample,rep, "../TEER/")
                
                # Remove duplicates and filter
                TEER_df_filt = teer_data.groupby('Timepoint').first().reset_index()  
                TEER_data_filt = TEER_df_filt [TEER_df_filt ["Timepoint"] == i]
                
                # Normalize TEER and drop Timepoint column
                TEER_data_filt.loc[:,"Rel. TEER"] = TEER_data_filt ["TEER"] /6.05
                TEER_data_filt = TEER_data_filt [["Rel. TEER"]]
                
                # Concatenate horizontally
                teer_sample_slice = pd.concat([teer_sample_slice, TEER_data_filt], axis=1)


       
            # Concatenate vertically (adds rows for this timepoint)
            logger_sample = pd.concat([logger_sample, logger_sample_slice])
            cluster_sample = pd.concat([cluster_sample, cluster_sample_slice])
            teer_sample = pd.concat([teer_sample, teer_sample_slice])

        # ======== Merge All Data Types ========
        # Rename TEER timepoint column to match logger/cluster "time" column
        teer_sample = teer_sample.rename(columns={"Timepoint": "time"})
        
        # Merge logger and cluster data on time
        full_logger_cluster_sample =  pd.merge(logger_sample,cluster_sample,how="outer",on="time")
        
        # Merge with TEER data
        full_logger_cluster_sample =  pd.merge(full_logger_cluster_sample,teer_sample,how="outer",on="time")
        
        # Drop time column 
        full_logger_cluster_sample = full_logger_cluster_sample.drop(["time"], axis=1)
        

        # ======== Combine Data from All Parameter Combinations ========
        # Concatenate 
        combined_full_logger_cluster_sample = pd.concat([combined_full_logger_cluster_sample,full_logger_cluster_sample] )

    # ======== Return Reshaped Array ========
    # Shape: (n_param_combinations, 5_timepoints, 50_features)
    return combined_full_logger_cluster_sample.to_numpy().reshape(-1,5,50)


# ============================================================================
# DATA PREPROCESSING AND CONFIGURATION
# ============================================================================

from numpy import inf

def configure_input(forward_dict):
    """
    Configure simulated data for training the BayesFlow model.
    
    This function performs several preprocessing steps:
    1. Log transformation 
    2. Handle invalid values (NaN, inf)
    3. Z-score normalization
    4. Parameter standardization

    
    """
    # Initialize output dictionary
    out_dict = {}

    # ======== Process Simulation Data ========
    # Apply log1p transformation: log(1 + x) to handle zeros gracefully
    logdata = np.log1p(forward_dict["sim_data"]).astype(np.float32) 
    
    # Optional: Add time encoding 
    #logdata [:,:,8] = np.linspace(0,1,logdata.shape[1])
    
    # Replace NaN values with -1 (sentinel value)
    logdata [:,:,: ]= np.nan_to_num (logdata [:,:,:], nan=-1)
    
    # Replace infinite values with -1
    logdata[logdata [:,:,:] == inf] = -1  
    
    # Z-score normalization using precomputed statistics from training data
    logdata = (logdata - obs_data_array_dict ["summary_conditions"].mean(axis=0)) / obs_data_array_dict ["summary_conditions"].std(axis=0)

    # ======== Process Parameters ========
    # Extract parameter values and convert to float32
    params = forward_dict["prior_draws"].astype(np.float32)
    

    params = ((params) - (prior_means)) / (prior_stds)

    # ======== Quality Control ========
    # Identify batches without any invalid values (NaN, inf, -inf)
    idx_keep = np.all(np.isfinite(logdata), axis=(1, 2))
    
    if not np.all(idx_keep):
        print("Invalid value encountered...removing from batch")

    # ======== Return Filtered Data ========
    # Only keep valid samples
    out_dict["summary_conditions"] = logdata[idx_keep]
    out_dict["parameters"] = params[idx_keep]

    return out_dict


# ============================================================================
# BAYESFLOW MODEL SETUP
# ============================================================================


# Construct save path based on hyperparameters

save_name = f"Results_Spatial_TimeSeqeunce/BF_XXX_SD_{SD}_CL_{CL}_DO_{DO}_LR_{LR}_Epochs_" + str(Epochs)

# Create simulator wrapper 
simulator = Simulator(batch_simulator_fun=partial(simulator_morpheus))

# Create generative model combining prior and simulator
model = GenerativeModel(
    prior, 
    simulator, 
    name="Simple_Infection_Generator2", 
    simulator_is_batched=True 
)


# ============================================================================
# NEURAL NETWORK ARCHITECTURE
# ============================================================================

summary_net = SequenceNetwork(summary_dim=int(SD)) #NEtwork architecture for summary network adapted from https://doi.org/10.1371/journal.pcbi.1009472

# Inference Network: 
inference_net = InvertibleNetwork(
    num_params=len(prior.param_names),  # 3 parameters: rho_adj, b_adj, cc_cf_ratio
    num_coupling_layers=int(CL),              # Number of coupling layers in the flow
    coupling_settings={
        'dropout_prob': float(DO),           # Dropout for regularization
        'bins': 32,                     # Number of bins for spline transformation
        'kernel_regularizer': tf.keras.regularizers.l2(1e-3)  # L2 regularization
    },
    coupling_design="spline"            # Use spline-based coupling layers
)

# Latent Distribution: Multivariate Student-t distribution
latent_dist = tfp.distributions.MultivariateStudentTLinearOperator(
    df=10,  # Degrees of freedom: higher values approach Gaussian, lower = heavier tails
    loc=np.zeros(inference_net.latent_dim, dtype=np.float32),  
    scale=tf.linalg.LinearOperatorDiag(
        np.ones(inference_net.latent_dim, dtype=np.float32)    
    ),
)

# Amortized Posterior: Combines summary and inference networks

amortizer = AmortizedPosterior(
    inference_net, 
    summary_net, 
    name="simple_amortizer2",
    latent_dist=latent_dist  # Use Student-t latent distribution
)

# Trainer: Manages the training process
trainer = Trainer(
    amortizer=amortizer,                # The amortized posterior to train
    generative_model=model,             # Generative model (prior + simulator)
    default_lr=float(LR),                 # Learning rate
    configurator=configure_input,       # Data preprocessing function
    memory=False,                       
    checkpoint_path=save_name           # Path to save model checkpoints
)

# Print network architecture summary
amortizer.summary()

# Load pre-generated offline training data. This can either include or exclude spatial data features.

with open('XXX_Offline_Data.pkl', 'rb') as f:
    offline_data = pickle.load(f)

# Prepare validation data during training 
# Take the last 500 samples from the offline data for validation during training
offline_data_testing = offline_data.copy()
for key in offline_data_testing.keys():
    offline_data_testing[key] = offline_data[key][-500:]

# ============================================================================
# TRAINING CONFIGURATION AND EXECUTION
# ============================================================================

# Setup optimizer with training parameters
trainer._setup_optimizer(
    optimizer=None,              # Use default Adam optimizer
    epochs=int(Epochs),          
    iterations_per_epoch=40      
)

# Store reference to optimizer
optimizer = trainer.optimizer

# Train the model using offline (pre-generated) data
history = trainer.train_offline(
    offline_data,                # Pre-simulated training data
    epochs=int(Epochs),          # Number of training epochs
    batch_size=256,              # Samples per batch
    
    # Validation data: ground truth parameters and their simulations
    validation_sims=offline_data_testing,
    
    early_stopping=True  # Stop if validation loss stops improving
)


# ============================================================================
# POST-TRAINING DIAGNOSTICS AND VISUALIZATION
# ============================================================================

# Plot training and validation losses
f = diag.plot_losses(
    history["train_losses"], 
    history["val_losses"], 
    moving_average=True  # Smooth curves with moving average
)
plt.savefig(save_name + "_Train_Loss_Plot.pdf")


import bayesflow as bf

# Prepare test data using the same configurator (preprocessing)
obs_data_array_testing = trainer.configurator({
    "sim_data": obs_data_array, 
    "prior_draws": np.array(testing_param_array)
})

# Sample from the learned posterior distribution
# Draw 1000 posterior samples for each test case
posterior_samples = amortizer.sample(obs_data_array_testing, n_samples=1000)

# Save posterior samples for further analysis
posterior_samples_filename = save_name + "_Posterior_Samples.pkl"
with open(posterior_samples_filename, 'wb') as f:
    pickle.dump(posterior_samples, f)

# ======== Diagnostic Plots ========

# 1. ECDF (Empirical Cumulative Distribution Function) plot
# Shows calibration: well-calibrated posteriors should show uniform ECDF
f = bf.diagnostics.plot_sbc_ecdf(
    posterior_samples, 
    obs_data_array_testing["parameters"], 
    difference=True  
)
plt.savefig(save_name + "_ECDF_Plot.pdf")

# 2. SBC (Simulation-Based Calibration) histograms
# Should be uniform if posterior is well-calibrated
f = bf.diagnostics.plot_sbc_histograms(
    posterior_samples, 
    obs_data_array_testing["parameters"],
    num_bins=20  # Number of histogram bins
)
plt.savefig(save_name + "_SBC_Plot.pdf")

# 3. Parameter Recovery plot
# Scatter plot: true vs. estimated parameters
# Points should lie on the diagonal for perfect recovery
f = bf.diagnostics.plot_recovery(
    posterior_samples, 
    obs_data_array_testing["parameters"]
)
plt.savefig(save_name + "_RecoveryPlot.pdf")

# ============================================================================
# END OF SCRIPT
# ============================================================================
