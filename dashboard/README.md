##

This repository contains two main components. 
1. the python script `process_data.py` which applys NLP techniques described below to a csv file of user comments placed in the `input` folder.
2. a quarto dashboard `quarto.qmd` which takes this processed data and displays it suitable for a non-technical audience.


## Setup

Python packages specified in `requirements.txt`

Quarto version >= 1.4 is required
R Packages to be installed are as follows
* shiny
* reticulate
* ggplot2
* dplyr
* glue

## Data Processing

Using the file `process_data.py` load in your free-text data of choice as a list, 
and modify any required hyperparameters. Then, run this file to process data to be used in the dashboard.

## Use

Once data processing has been done, run the file `quarto.qmd` to open the dashboard. Here, if multiple topic numbers were
selected in processing, the number of topics can be changed. Then, flip through each topic to see summaries.