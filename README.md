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

## Citation
Please cite our work if you find it useful:
```
@inproceedings{yang-etal-2022-visualizing,
    title = "Visualizing the Obvious: A Concreteness-based Ensemble Model for Noun Property Prediction",
    author = "Yang, Yue  and
      Panagopoulou, Artemis  and
      Apidianaki, Marianna  and
      Yatskar, Mark  and
      Callison-Burch, Chris",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.45",
    pages = "638--655",
    abstract = "Neural language models encode rich knowledge about entities and their relationships which can be extracted from their representations using probing. Common properties of nouns (e.g., red strawberries, small ant) are, however, more challenging to extract compared to other types of knowledge because they are rarely explicitly stated in texts.We hypothesize this to mainly be the case for perceptual properties which are obvious to the participants in the communication. We propose to extract these properties from images and use them in an ensemble model, in order to complement the information that is extracted from language models. We consider perceptual properties to be more concrete than abstract properties (e.g., interesting, flawless). We propose to use the adjectives{'} concreteness score as a lever to calibrate the contribution of each source (text vs. images). We evaluate our ensemble model in a ranking task where the actual properties of a noun need to be ranked higher than other non-relevant properties. Our results show that the proposed combination of text and images greatly improves noun property prediction compared to powerful text-based language models.",
}
```
