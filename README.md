# SARS-ALIculture

## A multi-scale multi-cellular model for Air-Liquid Interface cultures

The provided code defines a multi-scale individual cell-based model to simulate and analyze within the pseudo-stratified epithelium of Air-Liquid-Interface (ALI) cultures of human airway epithelium. It relates to the following publication with details on the model structure and implementation:
* Raach B, Bundgaard N, Haase M, Starruss J, Sotillo R, Stanifer M, Graw F: **"Influence of cell type specific infectivity and tissue composition on SARS-
CoV-2 infection dynamics within human airway epithelium"**

## Description
The modelling environment simulates pseudo-stratified human airway epithelium considering the different cell types of basal, secretory and ciliated cells that are organized within structured layers. It considers the processes of cell proliferation and differentiation, as well as viral infection, replication and diffusion by accounting for cell-type specific effects and characteristics. The model is implemented as a cellular Potts model using Morpheus. 

## Implementation & Requirements
The model is implemented wihtin Morpheus (), using the powerful cellular Potts modelling framework that provides a multi-scale multi-cellular modelling environment. The model was developed within Morpheus version 2.2.6. The xml-file of the model contains the basic code for bronchial epithelium using the corresponding parameterization for tissue regneration and infection dynamics. 
