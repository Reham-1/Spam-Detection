# Spam Email Classification using Decision Trees

A machine learning project for classifying emails as **Spam** or **Not Spam** using Decision Tree Classifiers and the **Spambase** dataset.

The project explores feature normalization, overfitting control, and class balancing techniques, including **SMOTE**.

## Dataset

The project uses the **Spambase dataset**, which contains:

* 4,601 email samples
* 57 features
* Binary target: Spam / Not Spam
* 70% Training, 15% Validation, 15% Testing

## Experiments

The project includes several experiments:

* **Feature Normalization:** MinMaxScaler, StandardScaler, and RobustScaler
* **Overfitting Control:** Testing different Decision Tree hyperparameters such as `max_depth`, `min_samples_split`, `min_samples_leaf`, and `ccp_alpha`
* **Class Balancing:** Random Undersampling, Random Oversampling, and SMOTE

All experiments, code, visualizations, and detailed analysis are available in the Jupyter Notebook.

## Key Results

The final models were compared based on their performance on the test set:

| Metric         | SMOTE Model | Pruned Model |
| -------------- | ----------: | -----------: |
| Accuracy       |  **93.49%** |       90.16% |
| Spam Recall    |    **0.93** |         0.86 |
| Spam Precision |    **0.91** |         0.89 |
| Spam F1-Score  |    **0.92** |         0.87 |
| Tree Depth     |          28 |           10 |
| Leaves         |         239 |           70 |

The SMOTE-based model achieved higher test performance, while the pruned model produced a smaller and less complex tree.

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn
* Matplotlib
* Jupyter Notebook

## Team

* Reham Mobark
* Alya Almatroodi— [GitHub](https://github.com/AlyaIbraheem)
* Nadeen Alameer
* Aljori Aladaili

