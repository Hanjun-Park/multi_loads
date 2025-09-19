

def metric_calculation(model_name, y_true, y_pred, logger):
    """Calculates and prints RMSE and R^2 for each load."""
    logger.info(f"----- Evaluating Model: {model_name} -----")
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    num_loads = y_true.shape[1]
    
    overall_rmse = metrics.root_mean_squared_error(y_true, y_pred)
    overall_r2 = metrics.r2_score(y_true, y_pred)
    logger.info(f"Overall Performance:")
    logger.info(f"  - RMSE: {overall_rmse:.4f}")
    logger.info(f"  - R^2 Score: {overall_r2:.4f}")

    if num_loads > 1:
        logger.info("Per-Load Performance:")
        for i in range(num_loads):
            rmse = metrics.root_mean_squared_error(y_true[:, i], y_pred[:, i])
            r2s = metrics.r2_score(y_true[:, i], y_pred[:, i])
            logger.info(f"  - Load {i+1}: RMSE = {rmse:.4f}, R^2 = {r2s:.4f}")
    logger.info("-" * 50)
    return overall_rmse, overall_r2


# Usage
rmse, r2 = metric_calculation(f"{model_name} (k={k})", y_test_numpy, y_pred, logger)
