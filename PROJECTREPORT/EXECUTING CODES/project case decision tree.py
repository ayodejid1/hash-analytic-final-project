import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns


df = pd.read_csv(r'C:\Users\AYODEJI\.spyder-py3\problemcase.csv')
print(df)

df = df.rename(columns={'satisfaction_level': 'satisfaction', 
                        'last_evaluation': 'evaluation',
                        'number_project': 'projectCount',
                        'average_montly_hours': 'averageMonthlyHours',
                        'time_spend_company': 'yearsAtCompany',
                        'Work_accident': 'workAccident',
                        'promotion_last_5years': 'promotion',
                        'sales' : 'department',
                        'left' : 'turnover'
                        })
print(df)
# Check to see if there are any missing values in our data set
print(df.isnull().any())

# Get a quick overview of what we are dealing with in our dataset
print(df.head())

#data exploration
# The dataset contains 10 columns and 14999 observations
print(df.shape)

# Check the type of our features. 
print(df.dtypes)

# Looks like about 76% of employees stayed and 24% of employees left. 
# NOTE: When performing cross validation, its important to maintain this turnover ratio
turnover_rate = df.turnover.value_counts() / len(df)
print(turnover_rate)

# Display the statistical overview of the employees
t=df.describe()
print(t)

from sklearn import preprocessing
#creating labelEncoder
le = preprocessing.LabelEncoder()

# Convert these variables into categorical variables
df["dept"] = df["dept"].astype('category').cat.codes
df["salary"] = df["salary"].astype('category').cat.codes

#Correlation Matrix
corr = df.corr()
sns.heatmap(corr,annot=True,cmap='seismic',
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.title('Heatmap of Correlation Matrix')
plt.show()
print(corr)


#spliting features into independent and dependent variables
X= df.iloc[:,[0,1,2,3,4,5,6,7,8]].values
Y= df.iloc[:,9].values
print(df)

#split dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.76,test_size=0.24,random_state=1)



from sklearn.tree import DecisionTreeClassifier
#fitting decision tree to training set
classifier1=DecisionTreeClassifier(criterion='entropy', random_state=1)
classifier1.fit(X_train, Y_train)

#predicting test set result
Y_pred= classifier1.predict(X_test)
print(Y_pred)

var_prob= classifier1.predict_proba(X_test)
var_prob[0,:]

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,classifier1.predict(X_test)))
from sklearn.metrics import classification_report
print(classification_report(Y_test, classifier1.predict(X_test)))



## plot the importances ##
importances = classifier1.feature_importances_
feat_names = df.drop(['turnover'],axis=1).columns


indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12,6))
plt.title("Feature importances by DecisionTreeClassifier")
plt.bar(range(len(indices)), importances[indices], color='lightblue',  align="center")
plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical',fontsize=14)
plt.xlim([-1, len(indices)])
plt.show()

