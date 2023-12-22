import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("insurance.csv")


X = dataset.drop('charges', axis=1)
y = dataset['charges']

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)


#if you want to see the dataset shapes just uncomment those lines
# print("the whole dataset shape :" , dataset.shape)
# print("train dataset shape:", X_train.shape, y_train.shape)
# print("validation dataset shape:", X_val.shape, y_val.shape)
# print("test dataset shape:", X_test.shape, y_test.shape)