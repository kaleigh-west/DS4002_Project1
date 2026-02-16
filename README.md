# DS4002_Project1
## This repository contains the dataset, EDA, and logistic regression model used to classify our real versus fake news dataset. 
## Software and Platform
This project was developed and tested using the following  environment:

Operating System: macOS

Programming Language: Python

Developed Environment: Jupyter Notebook, Visual Studio Code

Required Python Packages:
- pandas
- numpy
- matplotlib
- scikit-learn
- seaborn
- jupyter
## Documentation
### Data Folder
&emsp;Data appendix.pdf\
&emsp;README.md with instructions on how to download raw dataset\
&emsp;headlines_clean.csv
### Output Folder
Data Appendix Folder\
&emsp;headline_labels.png\
&emsp;headline_length_boxplot.png\
&emsp;headline_length_dist.png\
EDA Folder\
&emsp;Distribution_of_Headline_Lengths.png\
&emsp;Distribution_of_Subjects.png\
&emsp;Distribution_of_Subjects_Fake.png\
&emsp;Distribution_of_Subjects_Real.png\
&emsp;Distribution_of_Text_Lengths.png\
&emsp;Fake_Articles_by_Month.png\
&emsp;Fake_Headline_Length_Distribution.png\
&emsp;Fake_Text_Length_Distribution.png\
&emsp;Real_Articles_by_Month.png\
&emsp;Real_Headline_Length_Distribution.png\
&emsp;Real_Text_Length_Distribution.png\
Model Folder\
&emsp;confusion_matrix.png
### Scripts Folder
&emsp;EDA.py\
&emsp;dataappendix.py\
&emsp;logistic_regression.py
## Reproducibility
### Step 1: Download or Clone the Repository

Download this repository as a ZIP file or clone it:

```
git clone https://github.com/kaleigh-west/DS4002_Project1.git
cd DS4002_Project1
```

All commands below must be run from the project root directory: `DS4002_Project1`

---

### Step 2: Install Required Packages

This project was developed using Python 3.11.

Install dependencies:

```
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

---

### Step 3: Download the Original Dataset

The project uses the **Fake and Real News Dataset** from Kaggle:

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

After downloading, unzip the folder and place the following files into:

`DATA/`

Files required:
```
DATA/Fake.csv
DATA/True.csv
```

---

### Step 4: Run the Exploratory Data Analysis (EDA)

Run:

```
python SCRIPTS/EDA.py
```

This recreates the exploratory visualizations used in the report.

Pre-generated figures are also included in the `OUTPUT/` folder for reference.

---

### Step 5: Train the Model and Create the Analysis Dataset

Run:

```
python SCRIPTS/logistic_regression.py
```

This script will:

- Combine Fake.csv and True.csv
- Clean the headline text
- Create `DATA/headlines_clean.csv`
- Train the TF-IDF Logistic Regression classifier
- Output accuracy, precision, recall, F1 score, and confusion matrix

---

### Step 6: Generate the Data Appendix Outputs

Run:

```
python SCRIPTS/dataappendix.py
```

This script uses `DATA/headlines_clean.csv` to recreate the summary statistics and visualizations included in the Data Appendix.

---

### Output Files

All figures and outputs in the `OUTPUT/` folder are reproducible.  
They will be regenerated automatically when the scripts above are executed.
