from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
df = pd.read_csv(r"C:\Users\Naveen\Desktop\data sets\Iris (1).csv")
print(df)
df.isnull().sum()
x = df.iloc[:, 1:5].values
y = df.iloc[:, 5].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
knn = KNeighborsClassifier(metric="euclidean")
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print(y_pred)
print(knn.predict([[3, 4, 5, 2]]))
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
