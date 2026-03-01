from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load dataset safely
try:
    data = pd.read_csv("career_dataset.csv")
except:
    print("ERROR: career_dataset.csv file not found!")
    exit()

X = data.drop("Career", axis=1)
y = data["Career"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train models
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Choose best model
log_acc = accuracy_score(y_test, log_model.predict(X_test))
knn_acc = accuracy_score(y_test, knn_model.predict(X_test))

model = log_model if log_acc >= knn_acc else knn_model

career_dict = {
    0: "Data Scientist",
    1: "Web Developer",
    2: "Business Analyst",
    3: "UI/UX Designer",
    4: "Cybersecurity Analyst"
}

explanations = {
    0: "Strong analytical and programming skills.",
    1: "Good programming and technical interest.",
    2: "Strong communication and analytical skills.",
    3: "Creative and good communication ability.",
    4: "Strong analytical thinking and security interest."
}

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            user_data = pd.DataFrame([[ 
                float(request.form["cgpa"]),
                int(request.form["prog"]),
                int(request.form["comm"]),
                int(request.form["analytical"]),
                int(request.form["creativity"]),
                int(request.form["internship"]),
                int(request.form["interest"])
            ]], columns=["CGPA", "Prog_Skill", "Comm_Skill",
                        "Analytical", "Creativity",
                        "Internship", "Interest"])

            prediction = model.predict(user_data)
            probabilities = model.predict_proba(user_data)
            confidence = max(probabilities[0]) * 100

            result = career_dict[int(prediction[0])]
            reason = explanations[int(prediction[0])]

            return render_template(
                "index.html",
                result=result,
                confidence=round(confidence, 2),
                reason=reason
            )

        except Exception as e:
            return f"Error: {e}"

    return render_template("index.html")

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000, debug=True)