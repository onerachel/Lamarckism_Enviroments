# Lamarckian Inheritance Improves Robot Evolution in Dynamic Environments
We tested the Lamarckian system both in the real world and in a MuJoCo-based wrapper called Revolve2. The release version of Revolve2 used in this project is v0.3.8-beta1, available at https://github.com/ci-group/revolve2/releases/tag/v0.3.8-beta1. The setup for the physical experiments can be found here: https://github.com/onerachel/physical_robot_experiments.

Parameters used in the experiments are:
``` 
ENVIRONMENT = "FLAT" or "RUGGED" # type of environment
CHANGE_FREQUENCY = 0 or 2 or 5   # frequency of environmental changes during an evolutionary run
pop_size=50,
offspring_size=25,
nr_generations=30,
learner==RevDE,
learning_trials=280
``` 
## Installation 
Steps to install:
``` 
1. git clone https://github.com/onerachel/Lamarckism_Environments.git
2. cd Lamarckism_Environments
3. virtualenv -p=python3.8 .venv
4. source .venv/bin/activate
5. ./dev_requirements.sh
``` 
In case you have installation errors wrt multineat, find where the cereal library is installed, e.g.: find / -name "cereal*", then tell CPPNWIN where to look for cereal manually, e.g.: export CPATH=/usr/local/include

## Run experiments 
To run experiments
``` 
cd lamarckian_evolution
python optimize.py
``` 
To show the simulation, add --visualize: 
``` 
python optimize.py --visualize
``` 
To restart from the last optimization checkpoint, simply rerun: 
``` 
python optimize.py
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

## Highlighted results

## Documentation 

[ci-group.github.io/revolve2](https://ci-group.github.io/revolve2/) 
