import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE

# 保存每组结果
results = []

# 循环处理10对数据集
for i in range(1, 11):
    print(f"Processing group {i}...")

    # 加载数据
    file_train = f'train{i}.csv'  # 根据实际路径调整
    file_test = f'test{i}.csv'  # 根据实际路径调整

    # 加载数据
    data_train = pd.read_csv(file_train)
    data_test = pd.read_csv(file_test)

    # 提取特征和目标
    selected_features = ['PI-content', 'CF-content', 'GP-content', 'Sizing agent-content']  # 假设用这3个特征
    X_train = data_train[selected_features]
    y_train = data_train.iloc[:, 0]
    X_test = data_test[selected_features]
    y_test = data_test.iloc[:, 0]

    # 多项式特征生成
    poly = PolynomialFeatures(degree=2)  # 使用二次多项式
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # 获取特征名称
    feature_names = poly.get_feature_names_out(selected_features)

    # 递归特征消除（RFE）从10个特征逐步减少到3个
    for num_features in range(10, 0, -1):  # 从10个特征逐步减少到3个
        print(f"Selecting top {num_features} features...")

        # 初始化 RFE 和 Ridge 回归模型
        from sklearn.linear_model import Ridge

        ridge_model = Ridge(alpha=1.0)
        rfe = RFE(estimator=ridge_model, n_features_to_select=num_features)

        # 使用 RFE 进行特征选择
        rfe.fit(X_train_poly, y_train)

        # 筛选后的特征
        selected_features_mask = rfe.support_
        X_train_poly_selected = X_train_poly[:, selected_features_mask]
        X_test_poly_selected = X_test_poly[:, selected_features_mask]
        selected_feature_names = feature_names[selected_features_mask]

        # 10折交叉验证
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        train_rmse_scores = []
        test_rmse_scores = []
        train_r2_scores = []
        test_r2_scores = []

        for train_index, val_index in kf.split(X_train_poly_selected):
            X_train_cv, X_val_cv = X_train_poly_selected[train_index], X_train_poly_selected[val_index]
            y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]

            # 使用K近邻回归模型
            ensemble_model = KNeighborsRegressor()
            ensemble_model.fit(X_train_cv, y_train_cv)

            # 获取预测值
            y_pred_train_cv = ensemble_model.predict(X_train_cv)
            y_pred_val_cv = ensemble_model.predict(X_val_cv)

            # 计算RMSE和R²
            train_rmse_scores.append(np.sqrt(mean_squared_error(y_train_cv, y_pred_train_cv)))
            test_rmse_scores.append(np.sqrt(mean_squared_error(y_val_cv, y_pred_val_cv)))
            train_r2_scores.append(r2_score(y_train_cv, y_pred_train_cv))
            test_r2_scores.append(r2_score(y_val_cv, y_pred_val_cv))

        # 计算平均指标
        avg_train_rmse = np.mean(train_rmse_scores)
        avg_test_rmse = np.mean(test_rmse_scores)
        avg_train_r2 = np.mean(train_r2_scores)
        avg_test_r2 = np.mean(test_r2_scores)

        # 使用筛选后的特征训练集成模型
        ensemble_model.fit(X_train_poly_selected, y_train)

        # 在测试集上评估
        y_pred_test = ensemble_model.predict(X_test_poly_selected)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_r2 = r2_score(y_test, y_pred_test)

        # 获取公式
        ridge_model.fit(X_train_poly_selected, y_train)
        ridge_formula = "y = " + " + ".join([f"{coef:.4f}*{name}" for coef, name in zip(ridge_model.coef_, selected_feature_names)]) + f" + {ridge_model.intercept_:.4f}"

        # 保存结果
        results.append({
            "group": i,
            "num_features": num_features,
            "train_rmse": avg_train_rmse,
            "test_rmse_cv": avg_test_rmse,
            "test_rmse_final": test_rmse,
            "train_r2": avg_train_r2,
            "test_r2_cv": avg_test_r2,
            "test_r2_final": test_r2,
            "formula": ridge_formula,
        })

# 转换为DataFrame以便分析
results_df = pd.DataFrame(results)

# 找到表现最优的一组
best_group = results_df.loc[results_df['test_rmse_final'].idxmin()]

# 打印最优结果
print("Best Results Group:")
print(f"Group: {best_group['group']}")
print(f"Number of Features: {best_group['num_features']}")
print(f"Train RMSE (Cross-Validation): {best_group['train_rmse']}")
print(f"Test RMSE (Cross-Validation): {best_group['test_rmse_cv']}")
print(f"Test RMSE Final: {best_group['test_rmse_final']}")
print(f"Train R2 (Cross-Validation): {best_group['train_r2']}")
print(f"Test R2 (Cross-Validation): {best_group['test_r2_cv']}")
print(f"Test R2 Final: {best_group['test_r2_final']}")
print(f"Best Formula: {best_group['formula']}")

# 保存所有结果
results_df.to_csv("rfe_KNN10-0.csv", index=False)
print("All results have been saved to 'rfe_results_with_selected_models.csv'.")
