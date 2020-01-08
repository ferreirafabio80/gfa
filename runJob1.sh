#!/bin/bash
#$ -l tmem=4G
#$ -l h_vmem=4G
#$ -l h_rt=0:10:0
#$ -S /bin/bash
#$ -j y
#$ -N run_HCP

source /share/apps/source_files/python/python-3.7.2.source
python3 /home/fferreir/BayesianCCA/main.py --perc_miss 0.3 
