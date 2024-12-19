#!/bin/bash
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OMP_NUM_THREADS=2

python experimental_pipeline.py $1 $2 $3
