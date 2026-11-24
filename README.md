# Predicting employment type in India

Can household and job characteristics distinguish self-employed workers, regular salaried employees and casual labourers? This IIT Kanpur DMS672 project studies employed people aged 15 and above using India’s Periodic Labour Force Survey (PLFS), July 2020–June 2021.

## Project files

- [Analysis notebook](employment_classification.ipynb) — data preparation, exploratory analysis, feature engineering, model training and evaluation with saved outputs.
- [Group report](project_report.pdf) — survey population, modelling decisions, results and individual contributions.

## Reproduce the analysis

1. From the repository directory on macOS or Linux, run `cat data/plfs.zip.part-* > plfs.zip && unzip plfs.zip` to reassemble and extract the survey archive. On any platform, you can also run `python -c "from pathlib import Path; import zipfile, io; zipfile.ZipFile(io.BytesIO(b''.join(p.read_bytes() for p in sorted(Path('data').glob('plfs.zip.part-*'))))).extractall('.')"`.
2. In the notebook setup cell, set `DATA_DIR = "CSV_Unit_level_data_PLFS_July2020_June2021"`. This folder contains the four required CSVs for household and person records.
3. In a Python environment, run `python -m pip install jupyterlab numpy pandas matplotlib seaborn scipy scikit-learn`.
4. From the repository directory, run `jupyter lab employment_classification.ipynb` and execute the cells in order. The notebook writes validation predictions to `plfs_validation_predictions.csv`.

## Method and results

The notebook joins household attributes to person records and derives the three target classes from Current Weekly Status. It excludes earnings and status codes from the predictors to prevent target leakage and groups the train/validation split by household. Macro-F1 gives each employment class equal weight despite differences in class size.

On the reported validation split, soft voting across a random forest, histogram gradient boosting model and multilayer perceptron achieved **86.12% accuracy**, **85.64% macro-F1** and **96.27% macro one-vs-rest ROC-AUC**. These scores describe classification among employed survey respondents.

## Authors

Arsh Jaswal, Ayush Raj, Harshvardhan Agarwal, Pooja Kumari and Shilpee Kumari · Group 10, DMS672 Data Mining and Knowledge Discovery, IIT Kanpur.
