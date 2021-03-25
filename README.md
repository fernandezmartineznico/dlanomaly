# dlanomaly
Deep Learning methods for Anomaly Detection

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── prod           <- Production, canonical data sets for modeling.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         `1_model_<model-name>`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── src                <- Source code for use in this project. Scripts to train models and then use trained models to make
    │   │                     predictions
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts with models
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py


# Usage
cd src/
python3 pred_cvae.py (train_cvae.py for training the model)

# Development
Development is done in notebooks/, then refactored into src/

## Results
| | | | | | | | | | | | |
|-|-|-|-|-|-|-|-|-|-|-|-|
| | | | | | | | | | | | |
| |Model|Supervision|Objective|Activation|Input|Hidden Neurons /layer|Loss|Class weights|AUC|Recall @100|Recall @500|
| |Isolation Forest (baseline) {'max_samples': 131072}|Unsupervised|Anomaly score| |29| | | |0.954|43%|72%|
| |VAE|Unsupervised|Reconstruction|Tanh|29|20,10,5,10,20|MSE| |0.963|26%|53%|
| |Conditional VAE|Unsupervised|Reconstruction|lReLU|29|20,10,5,10,20|MSE| |0.938|28%|55%|
| | | | | | | | | | | | |
| |LR (baseline) {'newton-cg', 'l2', 'C':100}|Supervised|Classification| |29| |BCE| |0.976|80%|90%|
| |DNN|Supervised|Classification|ReLU|29|64, 64|BCE| |0.948|76%|90%|
| |DNN|Supervised|Classification|ReLU|29|64, 64|BCE|Balanced|0.983|72%|90%|
| |DNN Keras tuner|Supervised|Classification|ReLU|29|64, 96|BCE| |0.953|79%|90%|
| |DNN Autokeras|Supervised|Classification|ReLU|29|512, 1024|BCE| |0.933|**84%**|90%|
| | | | | | | | | | | | |
| |Conditional VAE latent|Semi Superised|Reconstruction + Classification|lReLU|5|20,10,5 => 64,64| |Balanced|0.978|66%|87%|
| |Conditional VAE + DNN merged|Semi Superised|Reconstruction + Classification|lReLU|[29, 5]|[64, 10], [1]| |Balanced|**0.99**|77%|89%|
