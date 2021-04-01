==============================
Challenge question's analysis:
==============================

Version id: (v20.01) Candidate ID: (C1882695) Initials: SM

Please read Capital_one_data_science_project_report.pdf for detail analysis of challenge's questions.


=====================
Description of files:
=====================
router.py: it creates a data directory and some subdirectory and paths for easy access. Add portability to the project.

dataset.py: Holds base Dataset Class. Subclasses need to override some methods to process and load data for
different dataset.

model_factory.py: Holds all models. Model expect a dataset to be passed.

pyutils.py: Different utility functions

question.py: Generate result for all the questions that are given in the challenges.


===========
How to run:
===========

- Recreate conda environment from conda_project_env.yml (This projects uses minimal number of common libraries such as
numpy, pandas, matplotlib, sciket-learn etc. May not need all the libraries from this environment)

- unzip the data and put transactions.txt under project directory

- Convert JSON data to CSV for faster load and create smaller dataset
    python main.py --which json_to_csv
    python main.py --which gen_small

- To regenerate all the data of challenges run
    python question.py

- Generate some data exploration plots and description for different dataset. Plots will be saved data/fig directory
    python main.py --which explore_data --what_data small

- Generate experimental data for all models
    python model_factory.py


=============
Unit testing:
=============
python -m unittest discover unittests


