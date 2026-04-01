import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pickle

# Load dataset
data = pd.read_csv("heart.csv")

# Count plot
sns.countplot(x="target", data=data)
plt.title("Heart Disease Distribution")
plt.show()

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

X = data.drop("target", axis=1)
y = data["target"]

# Predictions
y_pred = model.predict(X)

# Confusion Matrix
cm = confusion_matrix(y, y_pred)

sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
y_prob = model.predict_proba(X)[:,1]
fpr, tpr, _ = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
plt.plot([0,1],[0,1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()