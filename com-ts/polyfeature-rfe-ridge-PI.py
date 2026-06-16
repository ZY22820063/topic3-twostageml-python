# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE

# 保存每组结果
results = []

# 循环处理10对数据集
for i in range(1, 11):
    print(f"Processing group {i}...")

    # 文件路径
    train_file = f'train{i}.csv'  # 根据实际路径调整
    test_file = f'test{i}.csv'    # 根据实际路径调整
    
    # 加载数据
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    # 提取特征和目标
    selected_features = ['PI-content', 'CF-content', 'GP-content', 'Sizing agent-content']  # 假设用四个特征
    X_train = train_data[selected_features]
    y_train = train_data.iloc[:, 0]
    X_test = test_data[selected_features]
    y_test = test_data.iloc[:, 0]

    # 多项式特征生成
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # 获取特征名称
    feature_names = poly.get_feature_names_out(selected_features)

    # 将多项式特征添加到DataFrame中
    X_train_poly_df = pd.DataFrame(X_train_poly, columns=feature_names)
    X_test_poly_df = pd.DataFrame(X_test_poly, columns=feature_names)

    # 初始化 Ridge 回归模型
    ridge = Ridge(alpha=1.0, random_state=42)  # 调整 alpha 值以控制正则化强度

    # 递归特征消除（RFE）过程：从所有特征减少到最优子集
    # 请注意下面的 range(...) 需要根据需求进行修改
    # 这里示例为从 14 个特征（若小于 14 则取实际特征数）往下逐个减少
    start_features = min(len(feature_names), 10)  
    # 如果希望减到至少1个特征，请把 range(...) 第二个参数改成 0，再根据步长 -1
    # 例如：range(start_features, 0, -1)
    for num_features in range(start_features, 0, -1):
        print(f"Selecting top {num_features} features...")
        
        rfe = RFE(estimator=ridge, n_features_to_select=num_features, step=1)

        # 使用 RFE 进行特征选择
        rfe.fit(X_train_poly_df, y_train)

        # 筛选后的特征
        selected_features_mask = rfe.support_
        X_train_selected = X_train_poly_df.loc[:, selected_features_mask]
        X_test_selected = X_test_poly_df.loc[:, selected_features_mask]
        selected_feature_names = feature_names[selected_features_mask]

        # 留一法交叉验证
        loo = LeaveOneOut()
        train_r2_scores = []
        test_r2_scores = []
        train_rmse_scores = []
        test_rmse_scores = []

        for train_index, val_index in loo.split(X_train_selected):
            X_train_cv, X_val_cv = X_train_selected.iloc[train_index], X_train_selected.iloc[val_index]
            y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]

            # 拟合模型
            ridge.fit(X_train_cv, y_train_cv)

            # 预测并计算指标
            y_pred_train_cv = ridge.predict(X_train_cv)
            y_pred_val_cv = ridge.predict(X_val_cv)
            train_rmse_scores.append(np.sqrt(mean_squared_error(y_train_cv, y_pred_train_cv)))
            test_rmse_scores.append(np.sqrt(mean_squared_error(y_val_cv, y_pred_val_cv)))
            train_r2_scores.append(r2_score(y_train_cv, y_pred_train_cv))
            test_r2_scores.append(r2_score(y_val_cv, y_pred_val_cv))

        # 计算平均指标（LOO 得到的是 n 个结果，这里取均值）
        avg_train_rmse = np.mean(train_rmse_scores)
        avg_test_rmse = np.mean(test_rmse_scores)
        avg_train_r2 = np.mean(train_r2_scores)
        avg_test_r2 = np.mean(test_r2_scores)

        # 使用整个训练集重新训练模型
        ridge.fit(X_train_selected, y_train)

        # 提取公式
        coefficients = ridge.coef_
        intercept = ridge.intercept_
        # 这里设置了一个阈值 0.01 来过滤系数绝对值较小的特征
        formula = f"{intercept:.4f}" + "".join(
            [f" + ({coef:.4f})*{feature}" for coef, feature in zip(coefficients, selected_feature_names) if abs(coef) > 0.01]
        )

        # 在测试集上评估
        y_pred_test = ridge.predict(X_test_selected)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_r2 = r2_score(y_test, y_pred_test)

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
            "formula": formula
        })

# 转换为 DataFrame 以便分析
results_df = pd.DataFrame(results)

# 找到表现最优的一组
best_group = results_df.loc[results_df['test_rmse_final'].idxmin()]

# 打印最优结果
print("Best Group Results:")
print(f"Group: {best_group['group']}")
print(f"Number of Features: {best_group['num_features']}")
print(f"Train RMSE (Cross-Validation): {best_group['train_rmse']}")
print(f"Test RMSE (Cross-Validation): {best_group['test_rmse_cv']}")
print(f"Test RMSE (Final): {best_group['test_rmse_final']}")
print(f"Train R² (Cross-Validation): {best_group['train_r2']}")
print(f"Test R² (Cross-Validation): {best_group['test_r2_cv']}")
print(f"Test R² (Final): {best_group['test_r2_final']}")
print(f"Best Formula: {best_group['formula']}")

# 保存所有结果
results_df.to_csv("ridge_rfe_loo.csv", index=False)
print("All results have been saved to 'ridge_rfe_loo.csv'.")
