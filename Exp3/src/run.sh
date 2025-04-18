#!/bin/bash

dt_depth=3

python3 ./main.py --base_learner nb --ensemble_method none
python3 ./main.py --base_learner nb --ensemble_method bagging --num_base_learners 16
python3 ./main.py --base_learner nb --ensemble_method adaboost --num_base_learners 75

python3 ./main.py --base_learner dt --ensemble_method none --dt_depth $dt_depth
python3 ./main.py --base_learner dt --ensemble_method bagging --dt_depth $dt_depth --num_base_learners 16
python3 ./main.py --base_learner dt --ensemble_method adaboost --dt_depth $dt_depth --num_base_learners 75

