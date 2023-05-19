# Online-Fraud-Detection 
QUANTUM JANUARY COHORT FINAL PROJECT
Predicting Fraudulent Business Transactions at Blossom Bank.

Introduction:
Blossom Bank is a financial services institution offering a range of services, including pension management, investment banking, payment services, and retail and investment banking. The objective of this project is to evaluate at least two machine learning models to predict fraudulent business transactions. In this essay, we will outline the steps taken to build the models, including exploratory data analysis (EDA), feature engineering, model selection, training, and testing. We will also discuss the model evaluation results and provide recommendations based on the findings.

Exploratory Data Analysis (EDA):
To begin the project, we conducted a comprehensive EDA on the available dataset. We imported necessary Python libraries such as NumPy, Pandas, Seaborn, and Matplotlib. The dataset was read in CSV format using the read_csv() function, specifying the delimiter as ',' and encoding as 'ISO-8859-1'. The DataFrame was assigned the variable name Blossom_df.
During the EDA, we performed various data mining tasks, including checking the dataset type, shape, statistical description, and identifying null or missing data. To facilitate analysis, we created new DataFrames, namely Fradulent_Transactions, Type, and Blossom_cat. The analysis revealed that the dataset contained over one million data entries across ten categories. Among the five transaction types present (CASH_OUT, CASH_IN, TRANSFER, DEBIT, and ONLINE), only CASH_OUT and TRANSFER had fraudulent transactions. However, the proportion of fraudulent transactions was minimal, representing only 0.11% of the entire dataset.

We further visualized the interrelationships of the categorical variables using BarPlot, Heatmap, Pairplot, and Boxplot. The analysis highlighted that the median of fraudulent transactions was relatively small, although outliers were present in the dataset. Future updates to the analysis will involve a more in-depth investigation of the outliers and renaming categorical variables for readability. We also attempted to isolate the source of fraudulent transactions using 'nameOrig' and 'nameDest' variables, but the dataset did not provide clear separation.

Feature Engineering:
To carry out feature engineering on the DataFrame, we needed to perform encoding. Due to system capacity constraints, we decided to drop two categorical variables ('nameOrig' and 'nameDest') from the DataFrame and used the get_dummies() function to encode the remaining variables. We defined 'isFraud' as our target variable ('y') and the rest as predictors ('X'). In future updates, we plan to explore other encoding methods such as OneHotEncoder and LabelEncoder.

Model Selection, Training, and Testing:
After feature engineering, we split the data into training and test sets using the train_test_split() function from sklearn.model_selection. We selected two supervised machine learning models, RandomForestClassifier and DecisionTreeClassifier, to predict fraudulent transactions. Additionally, we utilized the Confusion Matrix to analyze the results and determine True Positives and False Negatives.

Model Evaluation and Recommendation:
After evaluating the models, we arrived at the following conclusions:

The RandomForestClassifier model outperformed the DecisionTreeClassifier model, with only two instances being incorrectly predicted as positive compared to 38 instances in the DecisionTreeClassifier model.

Both models exhibited similar accuracy, with RandomForestClassifier achieving 0.9998 and DecisionTreeClassifier achieving 0.9996.

Both models showed equal ability to identify positive instances, as evidenced by the same Recall and f1 scores.

Based on these findings, we recommend using the RandomForestClassifier model for predicting fraudulent business transactions due to its superior performance and accuracy.


INTERPRETING CONFUSION MATRIX

True Positive (TP): The number of instances that were actually positive and were correctly predicted as positive.
False Negative (FN): The number of instances that were actually positive but were incorrectly predicted as negative.
False Positive (FP): The number of instances that were actually negative but were incorrectly predicted as positive.
True Negative (TN): The number of instances that were actually negative and were correctly predicted as negative.
Interpreting the confusion matrix allows you to gain insights into the performance of your classification model:

Accuracy: It is the overall correctness of the model and can be calculated as (TP + TN) / (TP + TN + FP + FN). It measures the proportion of correctly classified instances.

Precision: It indicates the proportion of correctly predicted positive instances out of the total instances predicted as positive and can be calculated as TP / (TP + FP). It measures the model's ability to avoid false positives.

Recall (Sensitivity or True Positive Rate): It indicates the proportion of correctly predicted positive instances out of the total actual positive instances and can be calculated as TP / (TP + FN). It measures the model's ability to identify positive instances.

Specificity (True Negative Rate): It indicates the proportion of correctly predicted negative instances out of the total actual negative instances and can be calculated as TN / (TN + FP). It measures the model's ability to identify negative instances.

F1 Score: It is the harmonic mean of precision and recall and provides a balanced measure of both. It can be calculated as 2 * (Precision * Recall) / (Precision + Recall).

Analyzing the values in the confusion matrix and the derived metrics helps you understand the strengths and weaknesses of your classification model, identify any patterns of misclassification, and make improvements if necessary

 
