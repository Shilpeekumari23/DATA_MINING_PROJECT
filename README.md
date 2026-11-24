# Classifying employment in India with the Periodic Labour Force Survey

A construction worker can run a small business, earn a regular salary, or take casual wage work. How much can we infer about that employment arrangement from the worker’s occupation, hours, and household circumstances?

We studied this question using India’s **Periodic Labour Force Survey (PLFS), July 2020–June 2021**. Across 13 classifiers, a soft-voting ensemble achieved **85.64% macro-F1** on held-out households. Its most frequent error was confusing self-employment with regular salaried work.

[Explore the analysis notebook](employment_classification.ipynb) · [Read the six-page project report](project_report.pdf)

## From survey records to a classification task

**1. Identify the working population.** We joined household and person files from first visits and revisits, starting with 923,985 person records. We kept employed respondents aged 15 or older and mapped their Current Weekly Status to three classes. Since the survey revisits households, the counts below represent observations across visits.

| Employment arrangement | Survey records | Share of modelling sample |
|---|---:|---:|
| Self-employed | 149,584 | 47.7% |
| Regular salaried | 119,019 | 37.9% |
| Casual labour | 45,156 | 14.4% |
| **Total** | **313,759** | **100.0%** |

**2. Find what distinguishes the groups.** In the sample, 81.0% of agricultural workers were self-employed, while 73.0% of construction workers were casual labourers. Industry and occupation showed the strongest associations with employment type. We combined them with features such as schooling, weekly hours, rural or urban location, and household expenditure to build a set of 19 predictors. Earnings helped us verify the labels and were excluded from training because the type of earnings can reveal the answer.

**3. Evaluate on unseen households.** More than 43,000 households appeared in both first-visit and revisit files. We kept every observation from a household on the same side of the training/validation split. We ranked models by **macro-F1**, which gives each class equal weight. This makes performance on casual labour visible even though it accounts for only 14.4% of the sample.

## What the models learned

Selected results on **63,611 validation records**. Casual-labour recall measures the percentage of actual casual-labour records the model identified correctly.

| Model | Accuracy | Macro-F1 | Casual-labour recall |
|---|---:|---:|---:|
| Logistic regression | 80.01% | 79.27% | 85.20% |
| Pruned decision tree | 82.23% | 81.71% | 84.48% |
| Random forest | 83.69% | 83.13% | 85.61% |
| Histogram gradient boosting | 85.51% | 84.87% | **87.62%** |
| Multilayer perceptron | 85.35% | 84.92% | 80.44% |
| Soft-voting ensemble | **86.12%** | **85.64%** | 85.04% |

Averaging the class probabilities from the forest, boosting model, and neural network gave the highest overall macro-F1. For the narrower goal of identifying casual labourers, gradient boosting had the highest recall. The ensemble’s confusion matrix shows where ambiguity remained: it labelled 3,237 self-employed records as regular salaried and 2,690 regular salaried records as self-employed.

**Scope.** The model classifies an existing job using information about that job, including industry and occupation. The scores describe the unweighted 2020–21 survey sample. Performance in other years and nationally weighted performance remain open questions.

## Run the notebook

1. Clone or download this repository and open a terminal in its root folder.
2. Reassemble the seven files in [data/](data/) and extract the four survey CSVs. On macOS or Linux, run:

   ```bash
   cat data/plfs.zip.part-* > plfs.zip
   unzip plfs.zip
   ```

   On Windows, run this Python command from the repository folder:

   ```shell
   python -c "from pathlib import Path; import io, zipfile; parts = sorted(Path('data').glob('plfs.zip.part-*')); zipfile.ZipFile(io.BytesIO(b''.join(p.read_bytes() for p in parts))).extractall('.')"
   ```

3. Install the dependencies and open the notebook:

   ```bash
   python -m pip install jupyterlab numpy pandas matplotlib seaborn scipy scikit-learn
   jupyter lab employment_classification.ipynb
   ```

4. In the notebook’s setup cell, set `DATA_DIR = "CSV_Unit_level_data_PLFS_July2020_June2021"`. Run the cells in order to reproduce the analysis and generate `plfs_validation_predictions.csv`.

## Project team

Arsh Jaswal, Ayush Raj, Harshvardhan Agarwal, Pooja Kumari and Shilpee Kumari  
Group 10 · DMS672 Data Mining and Knowledge Discovery · IIT Kanpur

The [project report](project_report.pdf) includes individual contributions, the full model comparison, and the confusion matrix.
