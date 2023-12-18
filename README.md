# Fake-Advertisement
Analyze the data on how many people in a specific age groups are targeted by fake marketing ads based on browsing preferences. This algorithm helps in blocking unnecessary ads displaying above content and on web page.
Data source : kaggle
# Report 
ABSTRACT: 
Fake advertising refers to the practice of using deceptive or misleading information to promote a product or service. Logistic regression is a statistical modeling technique used to analyze the relationship between a set of independent variables and a binary outcome variable. In the context of fake advertising, logistic regression can be used to identify the factors that are most likely to contribute to the creation and dissemination of fake advertising
# INTRODUCTION: 
Logistic regression is a popular machine learning algorithm that is used for binary classification tasks, where the goal is to predict a binary outcome (e.g., yes/no or true/false). In advertising, logistic regression can be used to predict whether a user is likely to click on an ad or not based on their past behavior and demographic data.
To use logistic regression in advertising, the first step is to collect and preprocess data. This can include user behavior data such as clicks, impressions, and conversions, as well as demographic data such as age, gender, and location.
Next, the data is split into training and testing sets. The training set is used to train the logistic regression model, while the testing set is used to evaluate its performance.
Once the model is trained, it can be used to predict the probability of a user taking a particular action based on their behavior and demographic data. This probability can then be used to decide which ad to display to the user.
It's important to note that the use of logistic regression in advertising should be ethical and transparent. Advertisers should not use fake advertising or mislead users with false promises or claims. Additionally, advertisers should follow relevant laws and regulations, such as those related to data privacy and advertising practices.

# BLOCK DIAGRAM/INTERFACE DIAGRAM:
![image](https://github.com/harshithkonasani/Fake-Advertisement/assets/94668868/07b17224-4aa1-410b-9087-a804b387c0cd)
# APPROACH/METHODOLOGY:

Logistic regression is a statistical method that is commonly used for binary classification problems, where the goal is to predict a binary outcome (e.g., yes or no, true or false, etc.). In the context of fake advertising, logistic regression can be used to predict whether an advertisement is genuine or fake based on a set of features.
The first step in using logistic regression for fake advertising detection is to collect data. This data should include both genuine and fake advertisements, as well as a set of features that can be used to distinguish between the two. Examples of features that may be useful include the content of the advertisement, the source of the advertisement, and the timing of the advertisement.
Once the data has been collected, it can be split into a training set and a test set. The training set is used to train the logistic regression model, while the test set is used to evaluate its performance. During training, the logistic regression model will learn to assign a probability of being genuine or fake to each advertisement based on the features provided.
Once the model has been trained and evaluated, it can be used to predict whether new advertisements are genuine or fake. This can be done by feeding the features of the new advertisement into the model and obtaining a probability of being genuine or fake. Based on this probability, a decision can be made as to whether the advertisement should be approved or rejected.
# DATASET DETAILS:
This data set contains the following features:
•	'Daily Time Spent on Site': consumer time on site in minutes
•	'Age': cutomer age in years
•	'Area Income': Avg. Income of geographical area of consumer
•	'Daily Internet Usage': Avg. minutes a day consumer is on the internet
•	'Ad Topic Line': Headline of the advertisement
•	'City': City of consumer
•	'Male': Whether or not consumer was male
•	'Country': Country of consumer
•	'Timestamp': Time at which consumer clicked on Ad or closed window
•	'Clicked on Ad': 0 or 1 indicated clicking on Ad
# IMPLEMENTATION:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from google.colab import files 
uploaded = files.upload() 
d.read_csv('advertising.csv')
ad_data.head()
![image](https://github.com/harshithkonasani/Fake-Advertisement/assets/94668868/b0b28eaa-6ed4-4375-8b3a-95e634cf3e6f)
ad_data.info()
ad_data.describe()
![image](https://github.com/harshithkonasani/Fake-Advertisement/assets/94668868/f7f1ed6a-f1d8-43b9-81c5-2411a89fcc9e)
sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')
sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,color='red',kind='kde')
sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data,color='yellow')
sns.pairplot(ad_data,hue='Clicked on Ad',palette='bwr')
from sklearn.model_selection import train_test_split
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
LogisticRegression(C=1.0,class_weight=None,dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

![image](https://github.com/harshithkonasani/Fake-Advertisement/assets/94668868/157c6b2a-283b-46a4-b538-2b306a57dbcb)
![image](https://github.com/harshithkonasani/Fake-Advertisement/assets/94668868/764d91b8-9635-42ec-9d15-6e5234a8f6cb)
![image](https://github.com/harshithkonasani/Fake-Advertisement/assets/94668868/2d01bba4-c991-40c1-9e40-9f6e3d44eb06)
![image](https://github.com/harshithkonasani/Fake-Advertisement/assets/94668868/8b156b10-300b-4843-bdc5-de7bff6006f4)

![image](https://github.com/harshithkonasani/Fake-Advertisement/assets/94668868/a1423771-61ac-43f2-b064-cd0f0bacc375)


![image](https://github.com/harshithkonasani/Fake-Advertisement/assets/94668868/c085bd13-4923-49f7-8abe-4440c8d72bc2)

![image](https://github.com/harshithkonasani/Fake-Advertisement/assets/94668868/c8d436ae-6aa6-4f3a-aed8-77525698ab3c)

![image](https://github.com/harshithkonasani/Fake-Advertisement/assets/94668868/717cd33b-7725-4223-b1d1-1a2f70e34abf)

![image](https://github.com/harshithkonasani/Fake-Advertisement/assets/94668868/a25ca36c-f179-41f0-8899-05a09bd372ab)


# CONCLUSION:

In conclusion, it is not ethical or appropriate to use logistic regression or any other machine learning algorithm for fake advertising. Misleading or false advertising is not acceptable and can have serious consequences. Logistic regression is a statistical model used for binary classification problems and can be used for legitimate and ethical purposes such as spam classification, fraud detection, and medical diagnosis. It is important to adhere to ethical standards and to use machine learning for legitimate and ethical purposes.




