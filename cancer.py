import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# from sklearn.datasets import load_iris
from sklearn.datasets import load_diabetes


data =load_diabetes()

features = pd.DataFrame(data.data)

target = pd.DataFrame(data.target)

print(features,target)

X_train,X_test,y_train,y_test = train_test_split(features,target)

my_model = LogisticRegression()
my_model.fit(X_train,y_train)
y_pred = my_model.predict(X_test)
print(y_pred,y_test)


