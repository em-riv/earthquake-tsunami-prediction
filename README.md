# Earthquake & Tsunami Prediction ML Model

A machine learning project that predicts tsunami occurrences based on earthquake characteristics using ensemble methods and feature engineering.

## Project Overview

This project builds and compares multiple machine learning models to predict whether an earthquake will trigger a tsunami. The models analyze various seismic features including magnitude, depth, location, and ground shaking intensity to make predictions.

**Key Highlights:**
- Binary classification problem (Tsunami vs No Tsunami)
- Comparison of 4 ML algorithms with hyperparameter tuning
- Advanced feature engineering with interaction terms
- Comprehensive model evaluation with multiple metrics
- Deployed model with prediction API

## Problem Statement

Tsunamis are devastating natural disasters often triggered by underwater earthquakes. Early prediction can save lives by enabling timely evacuations. This project uses historical earthquake data to build predictive models that can assess tsunami risk based on seismic characteristics.

##  Dataset

**Features:**
- `magnitude`: Earthquake magnitude (Richter scale)
- `depth`: Depth of earthquake epicenter (km)
- `latitude/longitude`: Geographic coordinates
- `cdi`: Community Decimal Intensity (felt intensity)
- `mmi`: Modified Mercalli Intensity
- `sig`: Significance score

**Target Variable:**
- `tsunami`: Binary (0 = No Tsunami, 1 = Tsunami)

**Dataset Source:** 

https://www.kaggle.com/datasets/ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset

## Technologies Used

- **Python 3.13.5**
- **Data Analysis:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn
- **Model Persistence:** joblib

## Installation & Setup

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd Earthquake_tsunami
```

2. **Install required packages:**
```bash
pip install -r requirements.txt
```

3. **Run the notebook:**
```bash
jupyter notebook earthquake.ipynb
```

## Project Structure

```
Earthquake_tsunami/
│
├── earthquake.ipynb                          # Main analysis notebook
├── earthquake_data_tsunami.csv               # Dataset
├── earthquake_tsunami_optimized_model.pkl    # Trained model
├── earthquake_scaler.pkl                     # Feature scaler
├── README.md                                 # Project documentation
├── requirements.txt                          # Dependencies
└── .gitignore                                # Git ignore rules
```

## Methodology

### 1. Data Exploration & Analysis (EDA)
- Statistical analysis of earthquake features
- Distribution analysis by tsunami occurrence
- Correlation analysis and multicollinearity detection
- Outlier detection using box plots

### 2. Feature Engineering
- **Magnitude categories:** Low (<7.0), Medium (7.0-7.5), High (>7.5)
- **Depth categories:** Shallow (<70km), Intermediate (70-300km), Deep (>300km)
- **Interaction features:** `magnitude / (depth + 1)` to capture combined effects
- **Polynomial features:** `magnitude²` for non-linear relationships
- **Geographic features:** `abs(latitude)` for distance from equator

### 3. Model Training & Evaluation
Trained and compared 4 algorithms:
- Decision Tree Classifier
- Random Forest Classifier
- Logistic Regression
- Gradient Boosting Classifier

**Evaluation Metrics:**
- ROC-AUC Score (primary metric)
- Accuracy
- F1-Score
- Precision-Recall curves
- Confusion matrices

### 4. Hyperparameter Tuning
- Grid Search CV with 5-fold stratified cross-validation
- Optimized Random Forest parameters
- Learning curves for overfitting detection

### 5. Model Deployment
- Model saved using joblib
- Prediction function with feature engineering pipeline
- Test cases for validation

## Results

### Model Performance Comparison

| Model | ROC-AUC | Accuracy | F1-Score |
|-------|---------|----------|----------|
| **Optimized Random Forest** | **~0.95+** | **~0.97+** | **~0.85+** |
| Gradient Boosting | ~0.94 | ~0.96 | ~0.83 |
| Random Forest | ~0.93 | ~0.96 | ~0.82 |
| Decision Tree | ~0.87 | ~0.94 | ~0.75 |
| Logistic Regression | ~0.86 | ~0.93 | ~0.70 |


### Key Insights

1. **Shallow earthquakes** (depth < 70km) are significantly more likely to cause tsunamis
2. **Magnitude-depth interaction** is the most important engineered feature
3. **Random Forest** performed best due to its ability to capture non-linear relationships
4. **Geographic location** (proximity to coastlines/subduction zones) plays a crucial role
5. Models show good generalization with minimal overfitting after tuning

### Feature Importance

Top 5 most important features:
1. `mag_depth_interaction` - Combined magnitude and depth effect
2. `depth` - Earthquake depth
3. `magnitude` - Earthquake magnitude
4. `abs_latitude` - Distance from equator
5. `longitude` - Geographic longitude

## Usage Example

```python
# Load the model
import joblib
import pandas as pd

loaded_model = joblib.load('earthquake_tsunami_optimized_model.pkl')
loaded_scaler = joblib.load('earthquake_scaler.pkl')

# Make prediction
def predict_tsunami(magnitude, cdi, sig, depth, latitude, longitude):
    # Feature engineering (see notebook for complete implementation)
    # ...
    return prediction

# Example: High magnitude (7.8), shallow depth (15km)
result = predict_tsunami(
    magnitude=7.8, 
    cdi=5.0, 
    sig=850, 
    depth=15.0, 
    latitude=34.05, 
    longitude=-118.25
)

print(result)
# Output: {'prediction': 'Tsunami', 'tsunami_probability': '85%', ...}
```

## Future Improvements

- [ ] Incorporate real-time seismic data APIs
- [ ] Add temporal features (time of day, season)
- [ ] Include ocean depth and plate boundary distance
- [ ] Implement SHAP values for better model interpretability
- [ ] Deploy as web application with Flask/FastAPI
- [ ] Add ensemble stacking methods
- [ ] Collect more data for rare tsunami events (class imbalance)

## Key Learnings

1. **Domain knowledge matters:** Despite low correlation, magnitude and depth are crucial due to their interaction
2. **Feature engineering is powerful:** Engineered features outperformed raw features
3. **Ensemble methods excel:** Random Forest and Gradient Boosting significantly outperformed linear models
4. **Cross-validation prevents overfitting:** Stratified K-fold maintained class balance
5. **Multiple metrics needed:** Accuracy alone is insufficient for imbalanced datasets

## Author

**Emmanuel Rivera**
- GitHub: [@em-riv](https://github.com/em-riv)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Earthquake data from [Ahmed Mohamed Zaki on Kaggle](https://www.kaggle.com/datasets/ahmeduzaki/global-earthquake-tsunami-risk-assessment-dataset)
- Inspired by tsunami early warning systems
- Thanks to the scikit-learn community

---

If you found this project useful, please consider giving it a star!
