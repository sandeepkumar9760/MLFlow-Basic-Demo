# MLFlow-Basic-Demo

A comprehensive demonstration project showcasing MLFlow's capabilities for machine learning experiment tracking, model management, and hyperparameter tuning using the Wine Quality dataset.

## 📋 Project Overview

This project demonstrates how to use **MLFlow** to:
- Track machine learning experiments and their parameters
- Log metrics and models
- Compare multiple models
- Register and manage model versions
- Monitor training runs and visualize results

### Dataset
The project uses the **Wine Quality Dataset** from the UCI Machine Learning Repository. This dataset contains physicochemical properties of red wine and their quality ratings, making it perfect for regression tasks.

**Dataset Details:**
- Source: [UCI ML Repository](http://archive.ics.uci.edu/ml/datasets/Wine+Quality)
- Features: 11 physicochemical properties (fixed acidity, volatile acidity, citric acid, etc.)
- Target: Quality rating (0-10 scale)
- Samples: 1,599 wine samples

## 🎯 Models Implemented

This project compares the performance of 5 different regression models:

1. **Linear Regression** - Baseline linear model
2. **Support Vector Regression (SVR)** - Non-linear regression using support vectors
3. **Random Forest Regressor** - Ensemble method with 100 trees
4. **Decision Tree Regressor** - Single decision tree model
5. **ElasticNet** - Regularized linear regression combining L1 and L2 penalties

## 📊 Evaluation Metrics

Each model is evaluated using three key regression metrics:

- **RMSE (Root Mean Squared Error)** - Measures prediction error magnitude
- **MAE (Mean Absolute Error)** - Average absolute difference between predictions and actuals
- **R² Score** - Proportion of variance explained by the model (0-1 scale)

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/sandeepkumar9760/MLFlow-Basic-Demo.git
cd MLFlow-Basic-Demo
```

2. **Install required dependencies:**
```bash
pip install -r requirements.txt
```

This will install:
- `mlflow==2.2.2` - MLFlow framework for experiment tracking
- `boto3` - AWS SDK (optional, for cloud storage)

### Required Python Libraries

The following libraries are automatically installed:
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms and metrics
- **mlflow** - Experiment tracking and model management

## 🏃 Running the Project

### Option 1: Default Parameters

Run with default ElasticNet hyperparameters (alpha=0.5, l1_ratio=0.5):

```bash
python MLflow-Models.py
```

### Option 2: Custom ElasticNet Parameters

Run with custom alpha and l1_ratio values:

```bash
python MLflow-Models.py <alpha_value> <l1_ratio_value>
```

**Example:**
```bash
python MLflow-Models.py 0.1 0.3
```

This trains the ElasticNet model with alpha=0.1 and l1_ratio=0.3.

### Parameter Guide

- **alpha** (default: 0.5): Regularization strength. Higher values = stronger regularization
  - Range: 0.0 to 1.0
  - Typical values: 0.01, 0.1, 0.5, 1.0

- **l1_ratio** (default: 0.5): Balance between L1 and L2 penalties
  - 0.0 = Pure L2 (Ridge regression)
  - 1.0 = Pure L1 (Lasso regression)
  - 0.5 = Balanced ElasticNet

## 📈 Expected Output

When you run the script, you'll see:

```
===== LinearRegression =====
RMSE: 0.5823
MAE: 0.4567
R2: 0.3875

===== SVR =====
RMSE: 0.6234
MAE: 0.4891
R2: 0.3421

===== RandomForest =====
RMSE: 0.5345
MAE: 0.4123
R2: 0.4521

===== DecisionTree =====
RMSE: 0.6789
MAE: 0.5234
R2: 0.2876

===== ElasticNet =====
RMSE: 0.5945
MAE: 0.4678
R2: 0.3654
```

## 🔍 Viewing MLFlow UI

### Start the MLFlow Tracking Server

```bash
mlflow ui
```

Then open your browser and navigate to:
```
http://localhost:5000
```

### Using the MLFlow Dashboard

1. **Experiments Tab:** View all your experiment runs
2. **Compare Runs:** Side-by-side comparison of different models
3. **Metrics:** Visual comparison of RMSE, MAE, and R² scores
4. **Parameters:** View hyperparameters used for each run
5. **Models:** Access logged models and their metadata

## 📁 Project Structure

```
MLFlow-Basic-Demo/
├── MLflow-Models.py          # Main training script
├── requirements.txt          # Project dependencies
├── README.md                 # This file
└── mlruns/                   # MLFlow tracking directory
    └── 0/
        └── models/           # Trained model artifacts
```

## 🔧 Configuration

### Local vs Remote Tracking

By default, the project uses **local file-based tracking**. To use a remote MLFlow server:

```python
# Uncomment these lines in MLflow-Models.py
remote_server_uri = "http://<your-server-ip>:5000/"
mlflow.set_tracking_uri(remote_server_uri)
```

### Model Registration

Models are automatically registered when using a non-file tracking URI. The naming convention is:
- `LinearRegressionWineModel`
- `SVRWineModel`
- `RandomForestWineModel`
- `DecisionTreeWineModel`
- `ElasticNetWineModel`

## 📚 MLFlow Concepts

### Runs
Each model training is logged as a separate "run" with its own parameters and metrics.

### Experiments
All runs in this project belong to the default experiment and are organized under experiment ID "0".

### Artifacts
Trained model objects are saved as artifacts in the `mlruns/0/models/` directory.

### Parameters
Customizable values logged for each run (e.g., alpha, l1_ratio).

### Metrics
Quantitative performance measures logged after each run (RMSE, MAE, R²).

## 🎓 Learning Outcomes

By exploring this project, you'll learn:

- ✅ How to initialize and use MLFlow
- ✅ How to log parameters, metrics, and models
- ✅ How to track multiple models simultaneously
- ✅ How to compare model performance
- ✅ How to use the MLFlow UI for experiment analysis
- ✅ How to structure ML projects with experiment tracking
- ✅ Best practices for reproducible machine learning

## 🤝 Contributing

Feel free to:
- Add more models
- Experiment with different datasets
- Improve hyperparameter tuning
- Add cross-validation
- Implement feature engineering

## 📝 References

- [MLFlow Documentation](https://mlflow.org/docs/latest/index.html)
- [Wine Quality Dataset](http://archive.ics.uci.edu/ml/datasets/Wine+Quality)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Machine Learning Best Practices](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Sandeep Kumar**  
- GitHub: [@sandeepkumar9760](https://github.com/sandeepkumar9760)
- LinkedIn: [Connect with me](https://www.linkedin.com/in/sandeep-kumar-ds/)

---

**Happy Learning! 🚀**

If you found this project helpful, don't forget to star it! ⭐
