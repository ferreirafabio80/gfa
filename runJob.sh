#!/bin/bash
#$ -l tmem=4G
#$ -l h_vmem=4G
#$ -l h_rt=0:10:0
#$ -S /bin/bash
#$ -j y
#$ -N ABCD_small

source /share/apps/source_files/python/python-3.7.2.source
#$ -cwd
python3 "$*" 
