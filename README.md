# IRMM Project

### Instructions
To run the models, do the following:
1. clone the git repo
1. download the data from https://www.kaggle.com/competitions/predictive-maintenance/data
1. save the data to a directory which has the same name as "SOURCE_DIRECTORY" in the .env file. Also set a "TARGET_DIRECTORY". On my computer, I have
```
DATA_DIRECTORY=sequential
SOURCE_DIRECTORY=predictive-maintenance
RANDOM_STATE=42
```
1. install the requirements from requirements.txt
1. Run clean_data.py
1. Run prepare_data.py
1. Follow the instructions in the notebook report.ipynb