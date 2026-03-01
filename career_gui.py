import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("career_dataset.csv")

X = data.drop("Career", axis=1)
y = data["Career"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train two models
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Select best model
log_accuracy = accuracy_score(y_test, log_model.predict(X_test))
knn_accuracy = accuracy_score(y_test, knn_model.predict(X_test))

model = log_model if log_accuracy >= knn_accuracy else knn_model

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

# GUI Window
root = tk.Tk()
root.title("AI Career Recommendation System")
root.geometry("500x600")

def predict_career():
    try:
        user_data = pd.DataFrame([[
            float(entry_cgpa.get()),
            int(entry_prog.get()),
            int(entry_comm.get()),
            int(entry_analytical.get()),
            int(entry_creativity.get()),
            int(entry_internship.get()),
            int(entry_interest.get())
        ]], columns=["CGPA", "Prog_Skill", "Comm_Skill",
                     "Analytical", "Creativity",
                     "Internship", "Interest"])

        prediction = model.predict(user_data)
        probabilities = model.predict_proba(user_data)
        confidence = max(probabilities[0]) * 100

        result_label.config(
            text=f"Recommended Career: {career_dict[prediction[0]]}\n"
                 f"Confidence: {confidence:.2f}%\n"
                 f"Reason: {explanations[prediction[0]]}"
        )

    except:
        messagebox.showerror("Error", "Please enter valid values!")

# Labels and Entries
tk.Label(root, text="Enter Student Details", font=("Arial", 14)).pack(pady=10)

def create_field(label_text):
    tk.Label(root, text=label_text).pack()
    entry = tk.Entry(root)
    entry.pack()
    return entry

entry_cgpa = create_field("CGPA (0-10)")
entry_prog = create_field("Programming Skill (1-10)")
entry_comm = create_field("Communication Skill (1-10)")
entry_analytical = create_field("Analytical Skill (1-10)")
entry_creativity = create_field("Creativity Level (1-10)")
entry_internship = create_field("Internship (0=No,1=Yes)")
entry_interest = create_field("Interest (0=DS,1=Web,2=BA,3=UI/UX,4=Cyber)")

tk.Button(root, text="Predict Career", command=predict_career).pack(pady=15)

result_label = tk.Label(root, text="", font=("Arial", 12), fg="blue")
result_label.pack(pady=10)

root.mainloop()