Explanation for data:
#######################################################################

The files in each of the five folders are pickle files (.pkl) that contain the preprocessed raw data used to train and validate the neural posterior estimators in BayesFlow. The data were obtained by sampling parameter combinations from a predefined prior distribution and running eight repeated Morpheus simulations for each set of parameters sampled and using the corresponding Morpheus model. The Morpheus output was then processed to extract the required observational information. The use of disrupted sampling made it possible to incorporate realistic between-sample and between-timepoint variability, consistent with that observed in the experimental data.

Files with the including "population" only consist of bulk information. Files including spatial, include both bulk and image information.

Files called "offline_data" hold the simulated observational data & parameter combination of the training data in a ready-to-use BayesFlow format.
Files called "obs_data_array_X" hold only the simulated observational data of the validation data.
Files called "testing_param_array_X" hold only the parameter combinations used for the validation data.
Files called "obs_data_array_dict_X" hold the simulated observational data & parameter combination of the validation data in a ready-to-use BayesFlow format.

