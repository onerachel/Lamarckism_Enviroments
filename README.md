# Investigate Lamarckism in different enviroments
We tested Lamarckism in three different environments in real world and in a mujoco based wrapper called Revolve2. The release version of Revolve2 used in this project is v0.3.8-beta1 (https://github.com/ci-group/revolve2/releases/tag/v0.3.8-beta1).

Parameters used in the experiments are:
``` 
pop_size=50,
offspring_size=25,
nr_generations=30,
learner==RevDE,
learning_trials=280
``` 
## Installation 
Steps to install:
``` 
1. git clone https://github.com/ci-group/revolve2 --branch v0.3.8-beta1
2. cd revolve2
3. git clone git@github.com:onerachel/Lamarckism_Enviroments.git
4. virtualenv .venv
5. source .venv/bin/activate
6. ./dev_requirements.sh (first comment out "pip install -e ./runners/isaacgym[dev] && \")
``` 
In case you have installation errors wrt multineat, find where the cereal library is installed, e.g.: find / -name "cereal*", then tell CPPNWIN where to look for cereal manually, e.g.: export CPATH=/usr/local/include

## Run experiments 
To run experiments, e.g. lamarckian_point_navigation:
``` 
python optimize.py
``` 
To show the simulation, add --visualize: 
``` 
python optimize.py --visualize
``` 
To restart from the last optimization checkpoint, add --from_checkpoint: 
``` 
python optimize.py --from_checkpoint
``` 
To plot fitness:
``` 
python plot_fitness.py
``` 
To check the best robot wrt the fitness:
``` 
python rerun_best.py
```
To check the best robot wrt the fitness and save the video:
``` 
python rerun_best.py -r <OUTPUT-DIR>
```

## Examples


## Documentation 

[ci-group.github.io/revolve2](https://ci-group.github.io/revolve2/) 