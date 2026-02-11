# BayesHAE - Combining a multi-scale multi-cellular model of human airway epithelium with Bayesian inference using BayesFlow

## Model extensions and adaptations

* Lukas P, Gibeaud A, Schumer C, Arruda J, Guedj J, Terrier O, Graw F: **“Multimodal data integration to determine viral and innate immune kinetics in human airway epithelium”**

This publication shows how the model can be combined with **BayesFlow** (https://bayesflow.org/main/index.html), a framework for neural posterior estimation relying on simulation-trained neural networks, to allow parameter inference of viral and immune related processes. The publication contains further details on the applied changes to the model structure and implementation. Specific model adaptations include the following xml-files in the [Models folder](https://github.com/GrawLab/SARS-ALIculture/tree/main/Models) :

* **Model_HOM.xml:** Simulation of viral spread within a homogeneous tissue (2D mono-layer)
* **Model_HAE.xml:** Simulation of viral spread within pseudo-stratified human airway epithelium (w./o. innate immunity)
* **Model_HAE_Phi.xml:** Simulation of viral spread within pseudo-stratified human airway epithelium considering innate immune dynamics (adapted implementation compared to the original implementation above)
* **Model_HAE_Phi_star.xml:** Simulation of viral spread within pseudo-stratified human airway epithelium considering innate immune dynamics (adapted implementation compared to the original implementation above with fixed interferon production rate)
* **Model_HAE_Phi_star_SARSCoV2.xml:** Model used for analysing SARS-CoV-2 infection dynamics in experimental HAE culture systems

These model files have been used in combination with BayesFlow to test and validate the ability of parameter inference for viral and innate immune kinetics considering experimental-like data.

## Neural posterior estimation with BayesFlow

The files mentioned in the following contain the necessary information to allow neural posterior estimation with BayesFlow, as well as the data used for training within the publication mentioned above. 

The [Script](https://github.com/GrawLab/SARS-ALIculture/tree/main/Scripts) **BayesFlow_Training_and_Inference_HAE.py** contains the general BayesFlow architecture and workflow for the training and inference used to analyse the different HAE scenarios. Training was performed in BayesFlow-Version 1 using the standard offline fitting routine with an ADAM optimizer with the default cosine decay for the learning rate and a batch size of 256. The required training data were pre-simulated and stored on disk to avoid on-the-fly data generation with data provided in this GitHub-repository and explained below.The script can be adapted to each of the different models considered, with further details on the network architectures and hyperparameters used provided within the supplementary material of the manuscript indicated above.

The different folders within the [Data folder](https://github.com/GrawLab/SARS-ALIculture/tree/main/Data) (**M_HOM**, **M_HAE**, etc) contain the pre-processed raw data that were used to train and validate the neural posterior estimators. Data were generated as described within Text S1 of the article mentioned above. In brief, for each model, 10.000 parameter combinations were sampled for training from pre-informed prior distributions, with which the corresponding Morpheus model was simulated. The Morpheus output was then processed to extract the necessary spatial and bulk information. For each set of parameters, 8 individual repeats were run to generate a combined data set accounting for disrupted sampling, i.e., incorporating between-sample and between-timepoint variability (see manuscript for further details).

Within each folder, the individual files relate to:

* **offline_data** contains the simulated observational data & parameter combinations of the training data in a ready-to-use BayesFlow format.
* **obs_data_array_dict_X** contains the simulated observational data & parameter combination of the validation data in a ready-to-use BayesFlow format.
* **obs_data_array_X** only contains the simulated observational data of the validation data
* **testing_param_array_X** only contains the parameter combinations used for the validation data

In general, files containing the term "population" only comprise the bulk information (e.g. viral load, TEER, etc). Files containing the term “spatial”, comprise both, i.e., bulk and image information. Data files are provided as pickle files (.pkl).



