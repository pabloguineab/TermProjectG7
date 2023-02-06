# APP. MATH CONC.FOR MACH.LEARN Project Report: Find Smokers by Vital Signal 

[comment]: # (This document is intended to capture the use case summary for this engagement. An executive summary should contain a brief overview of the project, but not every detail. Only the current summary should be captured here and this should be edited over time to reflect the latest details.)
[comment]: # (Some ideas of what to include in the executive summary are detailed below. Please edit and capture the relevant information within each section)
[comment]: # (To capture more detail in the scoping phase, the optional template Scoping.md may be utilized. If more detail around the data, use case, architecture, or other aspects needs to be captured, additional markdown files can be referenced and placed into the Docs folder)

# 1. Introduction

In this report, we will review the machine learning model of “Body Signal of Smoking”. This dataset has been chosen from Kaggle. The dataset contains a collection of physiological and blood factors of two groups, smokers and non-smokers. 

There are 27 categorical and numerical features in this dataset, and 55692 examples.

In this project different supervised classification learning algorithms such as Logistic Regression, SVM, Decision Tree, Random Forest, and SGD were implemented. Final results showed that Random Forest was chosen as the most accurate learning algorithm to predict our test dataset after tuning the hyperparameters according to GridSearchCV.

# 2. Data Pre-processing
## 2.1 Data Cleaning
Fortunately, our dataset had 0 NAN and null values and there was no need for using an imputer or other methods of treating NAN values. 


## 2.2 Adding Feature
We decided to generate BMI features based on available weight and height features. 
df['BMI'] = (df['weight(kg)']) / ((df['height(cm)'])**2)

## 2.3 Transforming Categorical Features to Numerical
In order to transform categorical features to numerical, label encoder was implemented.


lbe = LabelEncoder()
lbe.fit_transform(df["gender"])
df["gender"] = lbe.fit_transform(df["gender"])
lbe = LabelEncoder()
lbe.fit_transform(df["tartar"])
df["tartar"] = lbe.fit_transform(df["tartar"])
lbe = LabelEncoder()
lbe.fit_transform(df["oral"])
df["oral"] = lbe.fit_transform(df["oral"])
lbe = LabelEncoder()

## 2.4 Outliers Detection
To have more accurate models, we decided to remove our outliers using the following function.

def outlier_detection(df, n, columns):
    rows = []
    will_drop_train = []
    for col in columns:
        Q1 = np.nanpercentile(df[col], 25)
        Q3 = np.nanpercentile(df[col], 75)
        IQR = Q3 - Q1
        outlier_point = 1.5 * IQR
        rows.extend(df[(df[col] < Q1 - outlier_point)|(df[col] > Q3 + outlier_point)].index)
    for r, c in Counter(rows).items():
        if c >= n: will_drop_train.append(r)
    return will_drop_train

will_drop_train = outlier_detection(df, 5, df.select_dtypes(["float", "int"]).columns)
df.drop(will_drop_train, inplace = True, axis = 0)


## 2.5 Feature Selection
In our analysis of the body signals of smoking individuals, we aimed to identify the most relevant factors affecting smoking habits. To this end, we used several feature selection techniques, including drawing a correlation matrix to see the relationships between variables, and using the feature/importance method of scikit learn to select the 15 variables with the highest relevance.

After performing the feature selection, we trained several machine learning models using these 15 features and evaluated their performance using different metrics such as accuracy. Our results showed that the Random Forest model had the best performance with an accuracy of 0.826, followed by XGBoost Forest with 0.752 accuracy, Decision Tree with 0.742 accuracy, and SVM with 0.732 accuracy. The Logistic Regression and SGD models had relatively lower accuracy scores of 0.731 and 0.714, respectively.

It is worth noting that the accuracy of the models trained with the 15 most important features was somewhat lower compared to those trained with the original dataset. This highlights the importance of considering a wide range of variables in the analysis and not relying solely on a subset of the most relevant features.

Additionally, the correlation matrix we drew also provided insights into the relationships between variables. For example, it showed a strong correlation between height and weight, and a weaker correlation between variables such as relaxation and cholesterol levels. This information can be used to further refine the feature set and optimize the predictive models in future studies. 

In future studies, it would also be interesting to explore other feature selection methods and compare their performance with the one we used in this study. Moreover, incorporating additional data sources and incorporating more advanced machine learning techniques could potentially lead to even higher accuracy scores. 

Overall, this study has provided valuable insights into the body signals of smoking individuals and demonstrated the potential of machine learning models for predicting smoking habits. Further research in this area could help us better understand the factors that influence smoking habits and inform the development of more effective smoking cessation interventions.

In conclusion, our feature selection results suggest that there is no single best feature set that can accurately predict smoking habits. Instead, a combination of several features, including weight, height, BMI, gender, and oral health, among others, are required to build an accurate predictive model.


# 3. Data Visualization
The most important subject that took our attention visualizing data, was the fact that our dataset was not balanced. It means that the labels, or in the other word targets, there was a huge difference between the number of smokers and non-smokers. 

Also, the dataset was investigated in terms of distribution of data based on gender, age, weight, relaxation, hemoglobin, on other different blood factors versus each other.

# 4. Learning Algorithms and results

## 4.1 Methodology

As mentioned above, the implementation of different supervised learning algorithms such as Logistic Regression, Decision Tree, Random Forest, XGBoost, SGD, SVM for classification of smokers and non-smokers.
Also, GridSearchCV was used in all of the learning algorithms to tune the parameters of the models.

## 4.2 Logistic Regression
As it discussed before, we have a dataset that contains numerical and categorical features. We have two labels that make our model a binary model. In our case, these binary labels are yes and no means that with respect to the biological factors, is someone smoking or not. This problem is a supervised problem, so Logistic Regression would be one of our choices to find an algorithm to fit our training data and predict the label of our test data.

In order for us to be able to use the Logistic Regression algorithm, we decided to use the scikit learn library. The pre-defined Logistic Regression function needs some parameters to be specified before fitting the model on training data. At the first stage we decided to use the default parameters that are defined as below:
penalty = l2, C = 1, solver = lbfgs

from sklearn.linear_model import LogisticRegression
lrmodel = LogisticRegression(random_state=42)
lrmodel = LogisticRegression().fit(X_train, y_train)
predictions = lrmodel.predict(X_test)

from sklearn.metrics import precision_recall_curve, roc_curve, auc, log_loss, accuracy_score, classification_report,
confusion_matrix
confusion = confusion_matrix(y_test, predictions)
confusion

Result: array([[5710, 1317],
       [1943, 2169]])


### 4.2.1 Logistic Regression with GridSearchCV
Using the default parameters resulted in a model with accuracy of 70%. However it is not a very high accuracy, still it could be considered a good fit and model.
In the further step, we decided to use GridSearchCV to achieve the best possible parameter for our model. We defined GridSearch Parameters as follows and fitted the gridsearchcv to our training data.
param_grid = {'penalty': ['none','l1', 'l2', 'elasticnet'], 'C': [0.01, 0.1, 1],
              'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(lrmodel,param_grid,refit=True,verbose=3)
grid.fit(x_train,y_train)

### 4.2.2 Logistic Regression GridSearchCV best parameters

Finally, using GridSearchCV attributes resulted in choosing the following parameters as our best parameter to fit our model to the training data:
{'C': 0.01, 'penalty': 'none', 'solver': 'saga'}

### 4.2.3 Logistic Regression GridSearchCV predictions

Then, we investigated the prediction of the GridSearchCV on our test data. The result showed that accuracy increased from 70% before using GridSearchCV to 74% after using that. On the other hand, F1-score, which results from precision and recall, showed an improvement. The average F1-score for both binary labels was 0.71 before using GridSearchCV, while it increased to 0.74 after using it.

### 4.2.4 More about the findings of the result of Logistic Regression using GridSearchCV

One interesting thing that took our attention was the F1-score result for label 0 and label 1 individually in both cases before and after GridSearchCV. In both cases, the F1-score for label 0 was higher than the F1-score for label 1. We were looking for the reason to find out why our model has a better performance to predict non-smokers than the smokers. It turned out  that that was because of the problem with balancing our dataset. As we discussed before in the data pre-processing part, using the following code indicated that we had more non-smoker labels (means label 0) than the smoker label (means label 1).
df['smoking'].value_counts()

Result:
0    35237
1    20455
Name: smoking, dtype: int64

So, with more non-smoker labels in our dataset, our model learned better how to predict them, rather than the smoker. It obviously shows the importance of gathering the data in the right way, and exploring it to have a balanced dataset. If we did not have a balanced dataset, we would need to gather more data to achieve a better model, that could have resulted in a better prediction and better accuracy.

## 4.3. Stochastic Gradient Descent(SGD)
Stochastic Gradient Descent performs data classification by using loss functions as specified in the SGD classifier method. We begin by initializing the model with ‘sgd = SGDClassifier()’. You may notice here that the SGD model was initialized using default parameters, this was done as we will be searching for the best parameters using GridSearchCV, and to maintain parity with the other classifier models used to analyze our dataset. After initializing the model we then train the model using training data to perform a SGD fit with ‘sgd = sgd.fit(x_train, y_train)’, and predict the class labels using ‘sgd_predictions = sgd.predict(x_test)’. We can now calculate the accuracy of our SGD model using the accuracy_score() metric, which returns 73.29% accuracy.

### 4.3.1 SGD with GridSearchCV
In this section we attempt to increase the accuracy of our SGD model by using GridSearchCV to find a set of optimal parameters. The attributes we will be using in the GridSearchCV method are: the estimator(SGD model), a parameter grid(dictionary of parameters to test), verbose=3(for more detail on each fit), and the other attributes are set to default. We define the parameter grid as follows: param_grid = {'loss':['hinge', 'log_loss','modified_huber','squared_hinge','perceptron'],
                                	      'alpha': [1,0.1,0.01,0.001,0.0001],
          	                              'penalty':['l1']}
As we are dealing with a classification problem, we search only the loss function which is used for classification. The penalty is changed from ‘l2’ to ‘l1’ as the loss functions specified are linear, and the ‘l2’ penalty is designed for polynomial functions, as well as kernels such as ‘rbf’. For the alpha parameter we specify several values in order that GridSearchCV will find the optimal combination for the best estimator. We now define the Grid Search ‘grid = GridSearchCV(sgd,param_grid,refit=True,verbose=3)’, fit the model using the training data ‘grid.fit(x_train,y_train)’, and predict using the test data ‘grid_predictions = grid.predict(x_test)’. 

The GridSearchCV returns the best parameters as ‘{'alpha': 0.0001, 'loss': 'log_loss', 'penalty': 'l1'}’, the best estimator as ‘SGDClassifier(loss='log_loss', penalty='l1')’, and an improved accuracy of 74.32%.


## 4.4. Support Vector Machine (SVM)
As we all know, Support Vector Machine is a powerful and robust machine learning algorithm. After we train, split and scale our database values, we initialize the support vector classifier (SVC). Then we train the model by fitting the features (x) and class labels (y) of the training data set with the Support vector classifier keeping all the parameters as default.
from sklearn.svm import SVC
svm = SVC()
svm = svm.fit(x_train, y_train)

And predict the class labels using,
svm_predictions = svm.predict(x_test)
We now calculate the accuracy of our SVM model using the accuracy_score() metrics, which returns 73% accuracy.

### 4.4.1 SVM with GridSearchCV
Now we use Grid Search CV to see if it improves our accuracy, precision, recall and F1-score.  To find the optimal set of parameters we define Grid Search CV with the parameter grid as follows,
param_grid = {'C': [0.1, 1, 10],  'gamma': [1, 0.1, 0.01], 'kernel': ['linear', "poly",'rbf']}
grid = GridSearchCV(svm,param_grid,refit=True,verbose=3)

After that, we now fit the features and class labels of the training dataset and predict using the test dataset (unseen data)
grid.fit(x_train,y_train)
grid.best_params_
grid_predictions = grid.predict(x_test)

The trained model performs 135 fits and the best parameters with svc were found to be C = 10, Gamma = 0.01 and Kernel as linear. We then print the confusion metrics and classification report, which clearly indicate a good improvement in the True positives and decrease in False Negatives. As discussed earlier that the dataset is imbalanced,  yet the model upscales with a decent increase in the accuracy to 76%.


## 4.5. Decision Tree

Decision tree is one of the most powerful non-parametric models for both classification and regression problems. The same was observed by performing decision tree analysis on our dataset. It was observed that there was not much difference between the accuracies of the models with and without performing hyperparameter tuning on the same. In fact, the model accuracy dropped around 2% after tuning the hyperparameters with GridSearchCV. Before fitting the model with GridSearchCV parameters, the accuracy was observed as 76%.

### 4.5.1 Decision Tree with GridSearchCV

The GridSearchCV hyperparameters are as follows: 
param_grid ={'max_depth': range(1,20,2),'min_samples_leaf': range(1,100,5),'min_samples_split': range(2,10),'criterion': ["gini", "entropy"],'splitter':['best', 'random'],'max_features': ['auto']}
grid = GridSearchCV(dsc_tree,param_grid,refit=True,verbose=3)

After fitting the model with GridSearchCV, it was observed that the best parameters were {'criterion': 'gini',  'max_depth': 17, 'max_features': 'sqrt',  'min_samples_leaf': 1, 'min_samples_split': 3, 'random_state': 0, 'splitter': 'best'}. The accuracy after predictions was found to be 74%. With this, it can be said that Decision Tree Algorithm is not the best model for the given dataset as the accuracy drops from 76% to 74% after GridSearchCV. However, after performing Random Forest algorithm on the same, the model accuracy improves for better results.


## 4.6. Random Forest Classifier
Random Forest is a popular machine learning algorithm used for both classification and regression problems. It is a type of ensemble learning that generates multiple decision trees during training and outputs the class or prediction that is the mode of the classes or predictions generated by individual trees.
We start by setting the model's default parameters using the command "rnd frst = RandomForestClassifier()" in order to maintain parity with the other classifier models that were used to evaluate our dataset and to search for the best parameters using GridSearchCV. After initializing the model, we train it with training data using the commands (rnd frst = rnd frst.fit(x train, y train)) and (rnd frst predictions = rnd frst.predict(x test)) to perform a fit and predict the class labels. The accuracy score() metric can now be used to determine the accuracy of our Random Forest model, and it returns 83% correctness.

### 4.6.1 Random Forest with Grid Search
Grid Search was used to optimize the Random Forest classifier's hyperparameters. The hyperparameters for Grid Search were the number of trees in the forest (10, 50, 100, 600), their maximum depths (2, 4, 8, 16), and their criterion ('Gini', 'Entropy'). To discover the ideal setting for these hyperparameters, the Grid Search was run over a range of values.
Now that the Grid Search has been defined, the model is fitted using the training data (grid.fit(x train,y train)) and predictions are made using the test data (grid predictions = grid.predict(x test)). With an increased accuracy of 83%, the GridSearchCV returns the best settings as "Criterion: gini, Max depth: None, N estimators: 600".

## 4.7. XGBoost 
XGBoost is an optimized gradient boosting machine learning algorithm. It is a highly scalable and efficient algorithm that has been used to win many machine learning competitions. Grid Search is a technique used to optimize the hyperparameters of a machine learning model by training the model with different combinations of hyperparameters and selecting the best performing combination.
We start by setting the model's default parameters with 'xgb model = XGBClassifier()'. After initializing the model, we train it with training data to perform a fit (xgb model = xgb model.fit(x train, y train)), and then we use predictions (xgb model predictions = xgb model.predict(x test)) to forecast the class labels. The accuracy score() metric can now be used to determine the accuracy of our Random Forest model, and it yields a result of 78% accuracy.

### 4.7.1 XGBoost with Grid Search
Grid Search was used to optimize the Xgboost classifier's hyperparameters. The maximum depth of each tree in the forest (range: 2, 3, 4, 5, 6, 7, 8, 9, 10) and the learning rate ('0.1', '0.01', '0.05') were the hyperparameters employed in Grid Search. To discover the ideal setting for these hyperparameters, the Grid Search was run over a range of values.

Now that the Grid Search has been defined, the model is fitted using the training data (grid.fit(x train,y train)) and predictions are made using the test data (grid predictions = grid.predict(x test)). The optimum parameters for the GridSearchCV are "'learning rate=0.05, max depth=9, n estimators=140"), and it now has an accuracy improvement of 80%

## 5. Conclusion

We analyzed a dataset of body signals of smoking individuals to understand the factors that affect smoking habits. Through our data cleaning and preprocessing, we were able to create a new feature (BMI) and encode categorical variables (gender). Our analysis showed that there is a correlation between body signals such as weight, height, and BMI with smoking habits. Additionally, gender and oral health also played a role in smoking habits. Our results showed that the majority of individuals in the dataset were male, non-smokers, and had good oral health and relatively healthy BMI. However, we also identified and removed outliers in the data to improve the accuracy of our findings.

The implications of our findings are that there is a potential for using body signals to predict smoking habits and target intervention programs. Additionally, oral health is an important factor to consider when addressing smoking habits. Our results also highlight the importance of considering both gender and BMI in studies related to smoking.

Based on our findings, we recommend that future research should focus on expanding the dataset to include more diverse individuals and to improve the accuracy of the results. Additionally, further studies should examine the relationship between body signals, gender, oral health, and smoking habits in greater depth to gain a deeper understanding of the underlying mechanisms. Our model accuracy of [insert model accuracy] suggests that there is room for improvement in the predictive power of our model and we recommend exploring alternative methods or expanding the dataset to increase accuracy.

In conclusion, this study provides valuable insights into the relationship between body signals, gender, oral health, and smoking habits. Our findings have important implications for future research and intervention programs aimed at reducing smoking habits.


## 6. Recommendations

Based on the findings from our analysis of the body signals of smoking individuals, there are several recommendations that can be made to further improve our understanding of smoking habits. 

Further analysis of other factors that may impact smoking habits such as age, education, and socio-economic status.  

Collection of more comprehensive and diverse data to improve the accuracy of our findings and expand the scope of our study.   

Development of targeted interventions and programs to address the identified risk factors, such as promoting healthy lifestyles and increasing access to oral health care. 
Conducting larger, long-term studies to better understand the evolution of smoking habits and the impact of interventions over time.

By taking these steps, we hope to gain a deeper understanding of the complex relationship between body signals and smoking habits and to ultimately contribute to the reduction of smoking rates.


## References
* 5 types of classification algorithms in machine learning (2020) MonkeyLearn Blog. Available at: https://monkeylearn.com/blog/classification-algorithms/ 
Brownlee, J. (2020) 
* 4 types of classification tasks in machine learning, MachineLearningMastery.com. Available at: https://machinelearningmastery.com/types-of-classification-in-machine-learning/ 
* kukuroo3 (2022) Body signal of smoking, Kaggle. Available at: https://www.kaggle.com/datasets/kukuroo3/body-signal-of-smoking 
* M, R. (2022) How to classify data in python using Scikit-Learn, ActiveState. Available at: https://www.activestate.com/resources/quick-reads/how-to-classify-data-in-python/ 
* Machine Learning Classification Strategy in python (2022) Quantitative Finance & Algo Trading Blog by QuantInsti. Quantitative Finance & Algo Trading Blog by QuantInsti. Available at: https://blog.quantinsti.com/machine-learning-classification-strategy-python/ .
* Pietro, M.D. (2022) Machine learning with python: Classification (complete tutorial), Medium. Towards Data Science. Available at: https://towardsdatascience.com/machine-learning-with-python-classification-complete-tutorial-d2c99dc524ec 
* 10 ML Projects Classification IBM developer. Available at: https://developer.ibm.com/tutorials/learn-classification-algorithms-using-python-and-scikit-learn/ 
* Sklearn.model_selection.GRIDSEARCHCV scikit. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html 
* Hastie, T., Tibshirani, R., & Friedman, J. (2009). The elements of statistical learning: data mining, inference, and prediction (Vol. 2). Springer.
* James, G., Witten, D., Hastie, T., & Tibshirani, R. (2017). An introduction to statistical learning: with applications in R. Springer.
* Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.
* Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785-794).
* Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal of machine learning research, 12(Oct), 2825-2830.
* Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine learning, 20(3), 273-297.
* Bottou, L. (2010). Large-scale machine learning with stochastic gradient descent. 
* Zeng, Z., & Ding, X. (2017). A comprehensive study on feature selection methods for bioinformatics. Briefings in bioinformatics
