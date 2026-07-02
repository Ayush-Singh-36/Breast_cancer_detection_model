#importing libraries & dependencies
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay # type: ignore
import pickle as pk
import matplotlib.pyplot as plt
import numpy as np

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
x = x.select_dtypes(include=np.number)

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

#Generate the matrix numbers
# y_test are the real labels, y_pred are your model's predictions
cm = confusion_matrix(y_test, y_pred)
print(cm)

#make it visual and easy to understand
labels = ["Benign", "Malignant"]
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Breast Cancer Predictor")
plt.show()


model_artifacts = {
    'model': model,
    'scaler': scaler
}

with open('breast_cancer_data.pkl', 'wb') as f:
    pk.dump(model_artifacts, f)

print("\nAll updated production artifacts saved successfully!")