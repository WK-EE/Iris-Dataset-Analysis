# Iris-Dataset-Analysis

#### Author: Wael Khalil

## Project Purpose
The purpose of this project is to gain insights on the Iris dataset using various algorithms implemented in the data science field.

Note:
This document is written in [Markdown](https://dillinger.io/). Use the link to render if needed. 
Most IDEs render markdown and it's currently a de facto standard in modern day software documentation.

To view this Markdown in the right format, open it in an IDE that supports Markdown. I used PyCharm CE.
PyCharm has a free version on their website if a download is needed: [PyCharm CE Download](https://www.jetbrains.com/pycharm/download/#section=mac)

---

## Project Breakdown
The purpose of this program is to perform some analysis on the iris dataset. 
This analysis involves our program:
- Demonstrating plots that allow us to visually see two sets of features and the
  class they belong to
- Sorting the dataset by feature
- Outlier detection and removal in each class
- Demonstrating plots of our detected outliers 
- Feature Ranking using two different techniques

---

## Running this Project
Note: This program is built using Python 3.7 or later. The IDE used while building it is PyCharm CE.

1. Download and install `Python version 3.7 or later` on your machine.
2. Navigate to the [iris_data_analysis]() directory
3. Run the program as a module with real inputs: `python3 -m iris_data_analysis`

The program outputs .xlsx files containing our sorted features. These .xlsx files will be written to
an output directory called 'Sorted_Data'. This directory is located inside the package of our 
program.

---

#### Project Usage:
```commandline
Usage: python3 -m iris_data_analysis

Positional Arguments: None

Optional Arguments: None
```

---

### Project Layout
* Iris-Dataset-Analysis/: 
  `This is the parent or "root" directory containing all the files below.`
    * iris.csv
      `The dataset we are working with.`
    * README.md
      `The guide you are currently reading.`
    * iris_data_analysis/: 
      `This is our module that holds all our python scripts and our entry point to the program.`
      * \__init\__.py 
        `This file is used to expose what functions, variable, classes, etc are exposed when scripts import this module`
      * \__main\__.py 
        `This file is the entrypoint to our program when ran as a program.`
