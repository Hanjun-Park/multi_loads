import numpy as np
from sklearn import metrics


def metric_calculation(model_name, y_true, y_pred):
    """
    Calculates
    RMSE and R^2 for each load + per-load.
    """
    
    print(f"Model: {model_name}")
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)


    num_loads = y_true.shape[1]
    
    overall_rmse = metrics.root_mean_squared_error(y_true, y_pred)
    overall_r2 = metrics.r2_score(y_true, y_pred)
    print(f"Overall:")
    print(f"RMSE: {overall_rmse:.4f}")
    print(f"R^2 Score: {overall_r2:.4f}")


    if num_loads > 1:
        print("Per-Load:")
        for i in range(num_loads):
            rmse = metrics.root_mean_squared_error(y_true[:, i], y_pred[:, i])
            r2s = metrics.r2_score(y_true[:, i], y_pred[:, i])
            print(f"Load {i+1}: RMSE= {rmse:.4f}, R^2= {r2s:.4f}")

    
    return overall_rmse, overall_r2


# Usage

model_name = 'DBSCAN'
k=5
y_test = np.array([[1,2,3],[2,3,4],[3,4,5]])
y_pred = np.array([[1.1, 1.9, 3.2],[2.2, 3.1, 3.8],[2.9, 4.2, 5.1]])

rmse, r2 = metric_calculation(f"{model_name} (k={k})", y_test, y_pred)
