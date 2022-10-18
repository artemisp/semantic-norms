# Visualizing the Obvious: A Concreteness-based Ensemble Model for Noun Property Prediction

## Environment instructions

Install the package dependencies 
```
pip install -r requirements.txt
```

## Data download

Follow the [instructions](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) to install the AWS CLI to download the data.

Run the download bash file from the projercts home directory 
```
bash download.sh
``` 

## Replicate experiments

To run experiments
```
python main.py --dataset concept_properties --model cem
```

The dataset options are `concept_properties`, `feature_norms`, `memory_color`

and the model options are `random`, `glove`, `ngram`, `bert`, `roberta`, `gpt2`, `gpt3`, `vilt`, `clip`, `cem`, `cem-pred`.


## Outputs

We provide the outputs of each model for each dataset in `outputs/output_{dataset_name}/`.

## Qualitative Analysis

We provide the code for qualitative analysis in the folder `qualitative_analysis`. 


## Eval
The implementation of evaluation metrics used for this paper can be found in `eval.py`. 