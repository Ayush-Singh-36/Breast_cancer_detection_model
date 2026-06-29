#importing libraries & dependencies
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#reading & reviewing data
server_df = pd.read_csv("data/breast-cancer.csv")
#server_df.info()

#creating a copy of the original dataset so that our model work as well our original dataset remain untouched
server2_df = server_df.copy()

#Drop the unnecessary 'id' column
if 'id' in server2_df.columns:
    server2_df = server2_df.drop(columns=['id'])

#seperating features and targets
x = server2_df.drop(columns = ['diagnosis'])
y = server2_df['diagnosis']

#encoding the object data type
labelencoder = LabelEncoder()
y_encoded = labelencoder.fit_transform(y)

#feature Scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#print(x_scaled[:6])
#print(y_encoded[:6])

#splitting the datasets into training and testing part
x_train, x_test, y_train, y_test = train_test_split(x_scaled, 
                                                    y_encoded, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    stratify=y_encoded
                                                    )

#print("x_train size ", x_train.shape)
#print("x_test size ", x_test.shape)
#print("y_train size ", y_train.shape)
#print("y_test size ", y_test.shape)

#initiating the model development
model = LogisticRegression(random_state=42)
print("Training the Logistic Regression model !!!")
model.fit(x_train, y_train)

#use the trained model to predict the test data
y_pred = model.predict(x_test)

#evaluating the model's performance
accuracy = accuracy_score(y_test, y_pred)

print("\n-- Model's Evaluation --")
print(f"Test accuracy {accuracy*100 :.2f}%")
print("\nDetailed Classification Report :- ")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))