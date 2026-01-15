# Fair Backward Compatibility (FBC)

This repository contains the official implementation of the **Fair Backward Compatibility** framework published in the paper *Fair Backward Compatibility: Definitions, Theoretical Framework, and Empirical Results*.

## Abstract
Machine learning model updates often prioritize aggregate performance (e.g., accuracy) while neglecting sample-wise behavior. This can lead to **negative flips**—instances where a new model ($f_{new}$) fails on samples correctly predicted by a legacy model ($f_{old}$). When these flips disproportionately affect groups defined by sensitive attributes (e.g., race or sex), the models become both backward-incompatible and unfair.

We propose **Fair Backward-Compatible Empirical Risk Minimization (FBCERM)**, a framework that integrates fairness-aware backward compatibility into modern ML algorithms. We provide theoretical risk bounds and demonstrate effectiveness through convex and differentiable relaxations across shallow and deep architectures.

## Key Definitions
* **Negative Flip (NF):** an instance where $f_{old}(x) = y$ (correct) but $f_{new}(x) \neq y$ (incorrect).
* **Backward Incompatibility:** the presence of negative flips during a model update.
* **Fair Backward Compatibility:** ensuring that the distribution of negative flips is not biased with respect to sensitive attributes.

## Mitigation Models
* **FBC-S**: joint optimization of B-ACC and FBC through multi-objective model selection 
* **FBC-D**: mitigation of FBC via the differentiable relaxation, also coupled with multi-objective model selection.
* **FBC-C**: mitigation of FBC via the convex relaxation, combined with multi-objective model selection. (SVM only)


## Repository Structure
The project is organized into two main modules, each containing its own data handling, configuration, and execution logic:

* **/shallow-fair-learning**: FBC implementation for traditional "shallow" models (SVM and XGBoost).
    * `main.py`:  entry point for running experimental sweeps across datasets and seeds.
    * `runner.py`: training of legacy models ($f_{old}$) and the application of FBC mitigation for new models ($f_{new}$).
    * `engines.py`: mathematical implementation of FBC-S, FBC-D, and FBC-C.
    * `config.py`: defines hyperparameter grids and experiment settings.
    * `utils/`: helper scripts for data preprocessing and calculation of metrics.

* **/deep-fair-learning**: FBC implementation for Neural Networks.
    * `main.py`: execution script for training and evaluating.
    * `cli.py`: command-line interface for managing experiment arguments, hyperparameters, and runtime settings.
    * `data_loader.py`: specialized data handling for the tested datasets.
    * `models/`: definition of Keras backbone architectures and custom differentiable loss functions that implement the FBCERM relaxations.
    * `training/`:  logic for the training loop and FBCERM relaxations.
    * `utils/`: helper scripts for data preprocessing and metrics.

**Note on Gurobi License**
The `FBC-C` engine in the shallow module requires a valid Gurobi license. If no license is detected, the script will skip `FBC-C` and run the remaining methods. More information in [Gurobi Licenses](https://www.gurobi.com/downloads/).

## Data Acquisition

Datasets are not hosted directly in this repository. Please download the datasets from the sources below and place them in the corresponding `data/` folders.

### Shallow Learning Datasets
The following tabular datasets are used in the `shallow-fair-learning` module. Most are available via the UCI Machine Learning Repository:

* **German Credit**: [Download from UCI](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
* **Arrhythmia**: [Download from UCI](https://archive.ics.uci.edu/ml/datasets/Arrhythmia)
* **COMPAS**: [Download from ProPublica GitHub](https://github.com/propublica/compas-analysis) (Use `compas-scores-two-years.csv`)
* **Adult**: [Download from UCI](https://archive.ics.uci.edu/ml/datasets/Adult)
* **Student Performance**: [Download from UCI](https://archive.ics.uci.edu/ml/datasets/Student+Performance)
* **Bank Marketing**: [Download from UCI](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

### Deep Learning Datasets
The `deep-fair-learning` module utilizes larger image and clinical datasets:
* **FairFace**: [Official GitHub Repository](https://github.com/joojs/fairface) (Contains links for the 108,501 racially balanced images).
* **UTKFace**: [Download from Project Page](https://susanqq.github.io/UTKFace/) (Aligned & Cropped version recommended).
* **Fitzpatrick 17k**: Access can be requested via the [official MIT library link](https://libraries.mit.edu/app/uploads/sites/20/2022/10/Fitzpatrick17kdataset.slides.10.28.22.pdf).
* **DDI (Diverse Dermatology Images)**: [Download from Stanford AIMI](https://ddi-dataset.github.io/index.html).
* **Marvel Character Dataset**: [Download from Kaggle](https://www.kaggle.com/datasets/amirdhavarshinis/marvel-characters).

### Setup Instructions
1. Create a `data/` directory inside both `/shallow-fair-learning` and `/deep-fair-learning`.
2. Ensure file names match the expected strings in `data_loader.py` or `config.py`.

## Installation & Setup
1. **Clone the repository:**
   ```
   bash
   git clone [https://github.com/AnnaPallares/fair-backward-compatibility.git](https://github.com/AnnaPallares/fair-backward-compatibility.git)
   cd fair-backward-compatibility
   ```

2. **Create the Environment:**
    ```
    bash
    conda env create -f environment.yaml
    conda activate fair-backward
    ```
3. **How to Run:**
Shallow Experiments
    ```
    bash
    cd shallow-fair-learning
    python main.py
    ```
Deep Learning Experiments
    ```
    bash
    cd deep-fair-learning
    python main.py
    ```

## **Citation**
If you use this code or framework in your research, please cite:

@article{pallares-lopez2026fbc,
  title={Fair Backward Compatibility: Definitions, Theoretical Framework, and Empirical Results},
  author={Pallares-Lopez, A. and Buselli, I. and Anguita, D. and Roli, F. and Oneto, L.},
  journal={Complex & Intelligent Systems (Special Issue), Springer Nature (Under Review)},
  year={2026},
  url={[https://github.com/AnnaPallares/fair-backward-compatibility](https://github.com/AnnaPallares/fair-backward-compatibility)}
}