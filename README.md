# Investigate Lamarckianism in different enviroments
We tested Lamarckianism in two different enviroments in real world and in a mujoco based wrapper called Revolve2. The release version of Revolve2 used in this project is v0.3.8-beta1 (https://github.com/ci-group/revolve2/releases/tag/v0.3.8-beta1).

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
1. git clone git@github.com:onerachel/Lamarckianism_Enviroments.git
2. cd Lamarckianism_Enviroments
3. virtualenv -p=python3.8 .venv
4. source .venv/bin/activate
5. ./dev_requirements.sh
``` 

## Run experiments 
To run experiments, e.g. lamarckian_point_navigation:
``` 
python lamarckian_evolution/optimize.py
``` 
To show the simulation, add --visualize: 
``` 
python lamarckian_evolution/optimize.py --visualize
``` 
To restart from the last optimization checkpoint, add --from_checkpoint: 
``` 
python lamarckian_evolution/optimize.py --from_checkpoint
``` 
To plot fitness:
``` 
python lamarckian_evolution/plot_fitness.py
``` 
To check the best robot wrt the fitness:
``` 
cd lamarckian_evoluation
python rerun_best.py
```
To check the best robot wrt the fitness and save the video:
``` 
cd lamarckian_evoluation
python rerun_best.py -r <OUTPUT-DIR>
```

## Examples


## Documentation 

[ci-group.github.io/revolve2](https://ci-group.github.io/revolve2/) 