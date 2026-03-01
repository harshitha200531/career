import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("career_dataset.csv")

# Separate features and target
X = data.drop("Career", axis=1)
y = data["Career"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# MODEL TRAINING
# ===============================

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
log_accuracy = accuracy_score(y_test, log_pred)

# KNN Model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)

print("\nLogistic Regression Accuracy:", log_accuracy)
print("KNN Accuracy:", knn_accuracy)

# ===============================
# SELECT BEST MODEL
# ===============================

if log_accuracy >= knn_accuracy:
    model = log_model
    print("\nBest Model Selected: Logistic Regression")
else:
    model = knn_model
    print("\nBest Model Selected: KNN")

# ✅ IMPORTANT: Create y_pred OUTSIDE if-else
y_pred = model.predict(X_test)

# ===============================
# CONFUSION MATRIX
# ===============================

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ===============================
# FEATURE IMPORTANCE (Only for Logistic)
# ===============================

if hasattr(model, "coef_"):
    print("\nFeature Importance (Weights):")
    for feature, coef in zip(X.columns, model.coef_[0]):
        print(feature, ":", round(coef, 3))

# ===============================
# USER INPUT PREDICTION
# ===============================

print("\nEnter Student Details for Career Prediction")

cgpa = float(input("Enter CGPA (0-10): "))
prog = int(input("Programming Skill (1-10): "))
comm = int(input("Communication Skill (1-10): "))
analytical = int(input("Analytical Skill (1-10): "))
creativity = int(input("Creativity Level (1-10): "))
internship = int(input("Internship Experience (0=No, 1=Yes): "))
interest = int(input("Interest Area (0=DS,1=Web,2=BA,3=UI/UX,4=Cyber): "))

user_data = pd.DataFrame(
    [[cgpa, prog, comm, analytical, creativity, internship, interest]],
    columns=[
        "CGPA",
        "Prog_Skill",
        "Comm_Skill",
        "Analytical",
        "Creativity",
        "Internship",
        "Interest",
    ],
)

prediction = model.predict(user_data)

probabilities = model.predict_proba(user_data)
confidence = max(probabilities[0]) * 100

career_dict = {
    0: "Data Scientist",
    1: "Web Developer",
    2: "Business Analyst",
    3: "UI/UX Designer",
    4: "Cybersecurity Analyst",
}

print("\nRecommended Career:", career_dict[prediction[0]])
print("Confidence Level: {:.2f}%".format(confidence))
explanations = {
    0: "You have strong analytical and programming skills suited for Data Science.",
    1: "Your programming ability and technical interest match Web Development.",
    2: "Your communication and analytical skills are ideal for Business Analysis.",
    3: "Your creativity and communication strength fit UI/UX Design.",
    4: "Your analytical thinking and security interest suit Cybersecurity."
}

print("Reason:", explanations[prediction[0]])

# ===============================
# MODEL COMPARISON GRAPH
# ===============================

models = ["Logistic", "KNN"]
accuracies = [log_accuracy, knn_accuracy]

plt.bar(models, accuracies)
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.show()