#!/bin/bash
#$ -l tmem=4G
#$ -l h_vmem=4G
#$ -l h_rt=35:0:0
#$ -S /bin/bash
#$ -j y
#$ -N run_HCP

source /share/apps/source_files/python/python-3.7.2.source
python3 /home/fferreir/BayesianCCA/main.py
