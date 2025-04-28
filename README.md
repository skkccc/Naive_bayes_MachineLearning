# Earthquake Damage Prediction Analysis Project

This is a project that uses the Naive Bayes algorithm to predict earthquake damage levels. The project employs machine learning methods to analyze seismic data and predict building damage levels.

## Environment Requirements

### Required Libraries
Before running the code, please ensure the following Python libraries are installed:
```
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
```

You can install all dependencies using the following command:
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

### Recommended IDEs
- PyCharm
- Visual Studio Code
- Jupyter Notebook

## Project Structure

```
.
├── naive_bayes_analysis.py    # Main program file
├── earthquake_damage.csv      # Original dataset
├── confusion_matrix.png       # Confusion matrix visualization
├── feature_importance.png     # Feature importance analysis plot
└── conclusion.txt            # Analysis conclusions
```

## File Descriptions

1. `naive_bayes_analysis.py`
   - Main program file
   - Contains data loading, preprocessing, model training, and evaluation functions
   - Implements SMOTE oversampling technique to handle data imbalance
   - Includes model performance evaluation and visualization features

2. `earthquake_damage.csv`
   - Original dataset file
   - Contains earthquake-related features and building damage level labels

3. Output Files
   - `confusion_matrix.png`: Visualization of the model's confusion matrix
   - `feature_importance.png`: Feature importance analysis plot
   - `conclusion.txt`: Analysis conclusions and model performance summary

## Running Instructions

1. Ensure all required libraries are installed
2. Place the `earthquake_damage.csv` file in the same directory as the Python script
3. Run the `naive_bayes_analysis.py` file
4. The program will automatically generate visualization results and analysis reports

## Notes

- Ensure the dataset file format is correct
- Sufficient memory space is required to process the dataset
- Python 3.7 or higher is recommended