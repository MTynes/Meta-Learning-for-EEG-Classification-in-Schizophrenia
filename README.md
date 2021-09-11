# Meta Learning for EEG Classification in Schizophrenia
 
 
This repository contains code for the feature extraction process and the experiments necessary to replicate the results from
the workshop paper Meta-Learning on Spectral Images of Electroencephalogram of Schizophrenics, previously accepted in the MeL4CV workshop at AAAI 2021.


The workshop paper explores the possibility of using popular meta learning models (MAML and Prototypical networks) to classify stacked spectrograms.
These spectrograms are representations of electroencephalogram (EEG) readings which were converted from continuous time-series to sets of images.
Specifically, two EEG datasets were used in the experiments; Dataset-1 was split into 20 second slices and Dataset-2 was split into 5-second slices. 
Samples were acquired with an overlap rate of 20% to provide more exhaustive and substantial input.
### Data Processing Pipeline
![Data Processing Pipeline](https://drive.google.com/uc?export=view&id=1_rYuQza2IhMOcZzisKKAakbY35yFGXxd)

A notebook for converting Dataset-2 EEG values to spectrograms, and a set of sample spectrogram images can be found in the files below:
   \Spectrogram Generators\Spectrogram_Generator_for_Dataset-2_(bio_msu_ru)_All_Channels_fractional_noverlap_SML.ipynb\
   \Data\Extracted\Sample_Spectrograms\



For both EEG datasets, MAML and prototypical networks were used to train and test the model.
In an additional set of experiments with MAML, pre-training was performed with miniImageNet, followed by further training with Dataset-2 and testing with Dataset-1 (and the reverse).

   /MLModels/Meta Learning Models/MAML_Pytorch_with_Dataset-1.ipynb\
   /MLModels/Meta Learning Models/MAML_Pytorch_with_Dataset-2.ipynb\
   /MLModels/Meta Learning Models/MAML_Pytorch_with_further_training_Test_Dataset-1.ipynb\
   /MLModels/Meta Learning Models/MAML_Pytorch_with_further_training_Test_Dataset-2.ipynb\
   /MLModels/Meta Learning Models/Prototypical_Networks_with_Dataset-1.ipynb\
   /MLModels/Meta Learning Models/Prototypical_Networks_with_Dataset-2.ipynb\

### Results
| Network Name                       | Val Loss | Val Acc | Test Acc | Test AUC | Test F1 Macro |
|------------------------------------|----------|---------|----------|----------|---------------|
| **Dataset-1**                      |          |         |          |          |               |
| CNN                                | 1.6223   | 0.5000  | 0.6646   | 0.8128   | 0.6468        |
| CNN + Fine Tuning                  | 0.6348   | 0.5455  | 0.5021   | 0.5014   | 0.3579        |
| Prototypical Networks + CNN        | 0.0112   | 0.9977  | 0.9768   | 0.9772   | 0.9768        |
| MAML + CNN                         | 0.5000   | 0.8945  | 0.8420   | 0.8426   | 0.8419        |
| MAML + CNN with Further Training   | 0.2184   | 0.9342  | 0.9683   | 0.9342   | 0.9341        |
|                                    |          |         |          |          |               |
| **Dataset-2**                      |          |         |          |          |               |
| CNN                                | 0.0000   | 1.0000  | 1.0000   | 1.0000   | 1.0000        |
| CNN + Fine Tuning                  | 0.5805   | 0.5014  | 0.6814   | 0.6892   | 0.6560        |
| Prototypical Networks + CNN        | 0.5980   | 0.8333  | 0.8587   | 0.8580   | 0.8584        |
| MAML + CNN                         | 1.317    | 0.7749  | 0.8133   | 0.8122   | 0.8097        |
| MAML + CNN with Further Training   | 0.1888   | 0.9541  | 0.9489   | 0.9489   | 0.9489        |



This repository contains additional feature extraction methods and ML experiments which were not included in the workshop paper.
The notebooks mentioned above are intended to be executed on Google Colab.

