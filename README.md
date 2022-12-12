# SARS-ALIculture

## A multi-scale multi-cellular model for Air-Liquid Interface cultures

The provided code defines a multi-scale individual cell-based model to simulate and analyze within the pseudo-stratified epithelium of Air-Liquid-Interface (ALI) cultures of human airway epithelium. It relates to the following publication with details on the model structure and implementation:
* Raach B, Bundgaard N, Haase M, Starruss J, Sotillo R, Stanifer M, Graw F: **"Influence of cell type specific infectivity and tissue composition on SARS-
CoV-2 infection dynamics within human airway epithelium"**

## Description
The modelling environment simulates pseudo-stratified human airway epithelium considering the different cell types of basal, secretory and ciliated cells that are organized within structured layers. It considers the processes of cell proliferation and differentiation, as well as viral infection, replication and diffusion by accounting for cell-type specific effects and characteristics. The model is implemented as a cellular Potts model using Morpheus. 

## Implementation & Requirements
The model is implemented within Morpheus (https://morpheus.gitlab.io/), using the powerful cellular Potts modelling framework that provides a multi-scale multi-cellular modelling environment. The model was developed within Morpheus version 2.2.6. The xml-file of the model contains the basic code for bronchial epithelium using the corresponding parameterization for tissue regneration and infection dynamics. 

## Movies
The repository contains supporting movies that accompany the aforementioned article. These include:

* **S1 Movie:** Simulation of growth of ALI-culture system for bronchial epithelium. Basal cells (green) are “overgrown” by secretory (blue) and ciliated (orange) cells over time leading to a pseudo-stratified epithelium. A reduced cell number is simulated with simulations started with a low number of cells (N~1265) to improve visualization of specific dynamics. Dynamics is followed over 100 days. Model implementation and parameterizations are given in detail in Materials & Methods.
* **S2-S5 Movie:** Simulation of SARS-CoV-2 infection dynamics in ALI-culture systems. Simulated infection dynamics within ALI-culture system of bronchial (**S2 Movie**) and nasal epithelium (**S3 Movie**) and bronchial epithelium exposed to 2.5% (**S4 Movie**) and 5% (**S5 Movie**) cigarette smoke extract. Simulations comprise a total of ~4.7x10^4 cells at the start and are followed over 25 days. Uninfected (green/blue/orange) and infected (brown/purple/red) cells are indicated.
