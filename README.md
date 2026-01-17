# Neural-Network-Based-House-Price-Prediction-A-Custom-Implementation
🎯 Quick Overview

This project implements a custom neural network from scratch (no TensorFlow, no PyTorch, no pretrained models) to predict house prices based on various features. The implementation focuses on preventing overfitting through advanced regularization techniques and scientific experimentation.
Key Highlights
✅ 100% Custom Implementation - Built using only NumPy

✅ No Pretrained Models - Everything coded from scratch
✅ Anti-Overfitting Focus - Early stopping, dropout, L2 regularization
✅ Manifold Learning - PCA and t-SNE for data exploration
✅ Scientific Approach - 6 different architectures tested
✅ Excellent Generalization - Train-Val gap < 0.05
🌍 SDG Alignment
SDG 11: Sustainable Cities and Communities
This project contributes to making cities inclusive, safe, resilient, and sustainable through:

🏘️ Affordable Housing: Accurate price predictions help identify fairly-priced properties
📊 Data-Driven Planning: ML insights assist urban development decisions
💡 Market Transparency: Provides objective price assessments
🤝 Accessibility: Helps homebuyers make informed decisions


✨ Features
Data Processing

Missing value imputation with intelligent strategies
Outlier detection and handling (IQR method)
Feature engineering with interaction terms
RobustScaler for better generalization

Manifold Learning

PCA Analysis - Dimensionality reduction and variance analysis
t-SNE Visualization - Non-linear pattern discovery
Feature Importance - Identify key price drivers

Neural Network

Custom activation functions (ReLU, Leaky ReLU, Tanh)
Multiple optimizers (SGD, Momentum, Adam)
Dropout regularization (0.3-0.5)
L2 regularization
Early stopping mechanism
Learning rate decay
Gradient clipping

Anti-Overfitting Techniques

✅ Early stopping with patience
✅ Dropout layers
✅ L2 weight regularization
✅ Smaller architectures
✅ Gradient clipping
✅ Learning rate decay


📁 Project Structure


week1_data_preprocessing.ipynb
week2_neural_network.ipynb
week3_hyperparameter_tuning.ipynb

Individual Components
Week 1: Data Preprocessing
python# Includes:
# - Dataset creation with missing values
# - EDA and visualizations
# - Manifold learning (PCA, t-SNE)
# - Feature engineering
# - Data scaling
Week 2: Neural Network Training
python# Includes:
# - Custom NN implementation
# - Forward/backward propagation
# - Early stopping
# - Model evaluation
# - Anti-overfitting techniques
Week 3: Hyperparameter Tuning
python# Includes:
# - 6 different architectures
# - Optimizer comparison
# - Regularization experiments
# - Best model selection
# - Comprehensive analysis

🔬 Technical Details
Neural Network Architecture
Best Model Configuration:
python{
    'Architecture': [n_features, 32, 16, 1],
    'Activation': 'ReLU',
    'Optimizer': 'Adam',
    'Learning Rate': 0.01 (with decay),
    'L2 Regularization': 0.01,
    'Dropout': 0.3,
    'Batch Size': 64,
    'Early Stopping': Patience 150
}
Activation Functions
ReLU (Primary)
f(x) = max(0, x)
f'(x) = 1 if x > 0, else 0
Leaky ReLU
f(x) = x if x > 0, else 0.01x
f'(x) = 1 if x > 0, else 0.01
Loss Function
Mean Squared Error with L2 Regularization
L = (1/m) Σ(y_pred - y_true)² + (λ/2m) Σ(w²)
Optimization
Adam Optimizer
m = β₁m + (1-β₁)∇L
v = β₂v + (1-β₂)(∇L)²
w = w - α * m̂ / (√v̂ + ε)

📊 Results
Performance Metrics
MetricTrainingValidationTestR² Score0.8830.8710.868RMSE (scaled)0.3420.3590.363RMSE (original)$28,450$29,820$30,150MAE (original)$21,300$22,100$22,450
Key Achievements

✅ Train-Val Gap: 0.017 (Excellent generalization!)
✅ Early Stopping: Activated at epoch 847/1500
✅ No Overfitting: Curves stay together
✅ Manifold Analysis: 6 components explain 95% variance

Comparative Analysis
ModelR²RMSE ($)GapStatusBalanced (Best)0.868$30,1500.017🏆 WinnerWide Shallow0.863$31,2000.023✅ GoodConservative0.851$32,8000.012✅ SafeModerate Depth0.859$31,5000.041⚠️ Slight overfitMomentum0.854$32,1000.028❌ Adam betterAggressive Reg0.842$33,9000.009❌ Too restrictive
