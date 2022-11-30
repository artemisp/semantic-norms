#!/bin/bash
#
#SBATCH --job-name=vilt
#SBATCH --output=/nlp/data/yueyang/prototypicality/semantic-norms/models/ViLT/res.txt
#
#SBATCH --ntasks=1
#SBATCH --time=6000:00
#SBATCH --mem-per-cpu=200
/nlp/data/artemisp/miniconda3/bin/python /nlp/data/yueyang/prototypicality/semantic-norms/models/ViLT/vilt_encoder.py