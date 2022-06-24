# Visualizing the Obvious: A Concreteness-based Ensemble Model for Noun Property Prediction

## Environment instructions

Install the package dependencies 
```
pip install -r requirements.txt
```

## Data download

Run the download bash file from the projercts home directory 
```
bash download.sh
```
This script will not download the raw images. To download the raw images for each of the datasets, use the links below:
* Concept Properties: [here](https://drive.google.com/uc?export=download&id=1KGAb31doDMIQCiqi3zg0464LF37kE7Xa)
* Feature Norms: [here](https://drive.google.com/uc?export=download&id=1TueIWc-sgZB3a6H9VFo_LQJ3dNCzgGm0)
* Memory Colors: [here](https://drive.google.com/uc?export=download&id=1dtBrTAbZbgpNP2O4BqPcgwPTVBBfBKdU)

and place them in `data/datasets/{dataset_name}/images/bing_images`. 

## Replicate experiments

To run experiments
```
python main.py --dataset concept_properties --model cem
```

The dataset options are `concept_properties`, `feature_norms`, `memory_colors`

and the model options are `random`, `glove`, `ngram`, `bert`, `roberta`, `gpt2`, `gpt3`, `vilt`, `clip`, `cem`, `cem-pred`.