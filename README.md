# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import pandas module and import the required data set.
2. find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection import train_test_split.
6. Assign the train dataset and test dataset.
7. From sklearn.tree import DecisionTreeClassifier.
8. Use criteria as entropy.
9. From sklearn import metrics.
10. find the accuracy of our model and predict the require values.


## Program:

/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Himavath M
RegisterNumber:  212223240053
*/
```
import pandas as pd
df = pd.read_csv("/content/Employee.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df['left'].value_counts())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['salary'] = le.fit_transform(df['salary'])

x = df[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]

print(x)
y = df['left']
print(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 30)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)
print("Y Predicted : \n\n",y_pred)

from sklearn import metrics

accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"\nAccuracy : {accuracy * 100:.2f}%")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
# df.head()
![image](https://github.com/user-attachments/assets/587dd1ad-3c4d-4047-820e-7ab280a8a81c)
# df.info()
![image](https://github.com/user-attachments/assets/4a103456-8371-457e-bf67-fdd800d74941)
# df.isnull().sum()
![image](https://github.com/user-attachments/assets/6d32b05b-c7ab-48e9-a71e-cad725742506)
# df['left'].value_counts()
![image](https://github.com/user-attachments/assets/f3e58dd1-0200-4d97-b074-585cea0c5ec2)
![image](https://github.com/user-attachments/assets/f5457fdc-66cb-4ad0-912d-9fb6775c2fbf)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
