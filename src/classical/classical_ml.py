import os
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Model setup
rf = RandomForestRegressor(n_estimators=100, random_state=42)
svr = SVR()
lasso = Lasso(random_state=42)
knn = KNeighborsRegressor()
voting = VotingRegressor(estimators=[
    ('rf', rf),
    ('knn', knn),
    ('svr', svr)
])

models = {
    "Random Forest": rf,
    "SVR": svr,
    "Lasso": lasso,
    "Voting Regressor": voting
}

DATA_DIR = "Data/features"
classes = ["Carbohydrate", "Lignin", "Lipid", "Protein", "Tannin", "Others", "Hydrocarbon"]
feature_cols = ["A_scaled", "B_scaled", "C_scaled", "D_scaled", "E_scaled"]
target_col = "CCS"

results = []

for class_name in classes:
    class_dir = os.path.join(DATA_DIR, class_name)
    train = pd.read_csv(os.path.join(class_dir, "train.csv"))
    test = pd.read_csv(os.path.join(class_dir, "test.csv"))
    X_train, y_train = train[feature_cols], train[target_col]
    X_test, y_test = test[feature_cols], test[target_col]

    for model_name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, preds)

        results.append({
            "Class": class_name,
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Train_Time": train_time,
            "Num_Train": len(X_train),
            "Num_Test": len(X_test),
        })
        print(f"{class_name} {model_name}: R2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}, Train_Time={train_time:.2f}s")

# Save results for further analysis or plotting
results_df = pd.DataFrame(results)
results_df.to_csv("classical_ml_results.csv", index=False)
print("✅ All classical models trained and evaluated per class.")

