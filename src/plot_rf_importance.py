import pandas as pd
import matplotlib.pyplot as plt

# Load Random Forest feature importance
df = pd.read_csv("results/rf_feature_importance.csv")

# Select the top 10 features
top10 = df.head(10).sort_values("importance")

# Create chart
plt.figure(figsize=(10, 6))
plt.barh(top10["feature"], top10["importance"])

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Random Forest Feature Importance")
plt.tight_layout()

# Save chart
plt.savefig("results/rf_feature_importance.png", dpi=300)
plt.show()

print("Saved -> results/rf_feature_importance.png")