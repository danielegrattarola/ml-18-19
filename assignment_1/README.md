# Assignment 1

In this assignment you have to solve a regression problem.  
You are given a set of data consisting of input-output pairs (x, y), and you have to build a model to fit this data.  
We will then evaluate the performance of your model on a **different test set**.

In order to complete the assignment, you have to address the tasks listed below and submit your solution as a zip file on the iCorsi platform. 


## Tasks

1. Use the family of models `f(x, theta) = theta_0 + theta_1 * x_1 + theta_2 * x_2 + theta_3 * x_1 * x_2` to fit the data:
    - report the values of the trained parameters `theta`;
    - evaluate the test performance of your model using the mean squared error as performance measure;
2. Consider any family of non-linear models of your choice to address the regression problem:
    - evaluate the test performance of your model using the mean squared error as performance measure;
    - compare your model with the linear regression of task 1. Which one is **statistically** better?

**Bonus**: in the [Github repository of the course](https://github.com/danielegrattarola/ml-18-19), you will find a trained Scikit-learn model that we built using the same dataset you are given. This _baseline_ model is able to achieve a MSE of **0.02**, when evaluated on the test set.
You will get extra points if the test performance of your model is **better** (i.e., the MSE is lower) than ours. Of course, you also have to tell us **why** you think that your model is better.

In order to complete the assignment, you must submit a zip file on the iCorsi platform containing: 

1. a PDF file describing how you solved the assignment, covering all the points described above (at most 2500 words, no code!);
2. a working example of how to load your **trained model** from file, and evaluate it;
3. the source code you used to build, train, and evaluate your model.

See below for more details.


## Tools

Your solution must be entirely coded in **Python 3** ([not Python 2](https://python3statement.org/)), using the tools we have seen in the labs.
These include: 

- Numpy
- Scikit-learn
- Keras

You can develop your code in Colab, like we saw in the labs, or you can install the libraries on your machine and develop locally.  
If you choose to work in Colab, you can then export the code to a `.py` file by clicking "File > Download .py" in the top menu.  
If you want to work locally, instead, you can install Python libraries using something like the [Pip](https://pypi.org/project/pip/) package manager. There are plenty of tutorials online. 


## Submission

In the [Github repository of the course](https://github.com/danielegrattarola/ml-18-19), you will find a folder named `assignment_1`.
The contents of the folder are as follows: 

- `data/`:
    - `data.npz`: a file storing the dataset in a native Numpy format;
- `deliverable/`:
    - `run_model.py`: a working example of how to evaluate our baseline model;
    - `baseline_model.pickle`: a binary file storing our baseline model;
- `src/`:
    - `utils.py`: some utility methods to save and load models;
- `report_surname_name.pdf`: an example report;

The `run_model.py` script loads the data from the data folder, loads a model from file, and evaluates the model's MSE on the loaded data.  
When evaluating your models on the unseen test set, **we will only run this script**.  
You cannot edit the script, except for the parts necessary to load your model and pre-process the data. Look at the comments in the file to know where you're allowed to edit the code. 

You must submit a zip file with a structure similar to the repository, but:

- the `deliverable` folder must contain:
    - `run_model.py`, edited in order to work with your models;
    - the saved models for both tasks (linear regression and the model of your choice);
    - any additional file to load the trained models and evaluate their performance using `run_model.py`;
- the `src` folder must contain all the source files that your used to build, train, and evaluate your models;
- the report must be completed;

The file should have the following structure: 
```bash
as1_surname_name/
    report_surname_name.pdf
    deliverable/
        run_model.py
        linear_regression.pickle  # or any other file storing your linear regression
        nonlinear_model.pickle  # or any other file storing your model of choice
    src/
        file1.py
        file2.py
        ...        
```
Remember that we will **only execute** `run_model.py` to grade your assignment, so make sure that everything works out of the box.


## Evaluation criteria

You will get a positive evaluation if:

- you demonstrate a clear understanding of the main tasks and concepts;
- you provide a clear description of your solution;
- you provide sensible motivations for your choice of model and hyper-parameters;
- the statistical comparison between models is conducted appropriately;
- your code runs out of the box (i.e., without us needing to change your code to evaluate the assignment);
- your code is properly commented;
- your model has a good test performance on the unseen data;
- your model has a better test performance than the baseline model provided by us;

You will get a negative evaluation if: 

- we realise that you copied your solution;
- the description of your solution is not clear, or it is superficial;
- the statistical comparison between models is not thorough;
- your code requires us to edit things manually in order to work;
- your code is not properly commented;
