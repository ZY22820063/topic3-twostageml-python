# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 15:35:47 2024

@author: 15297
"""

import pandas as pd
import numpy as np
import scipy
from sklearn.model_selection import LeaveOneOut  # 修改1：导入LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

print("XGBoost and SciPy are working correctly!")

# 保存每组结果
results = []

# 循环处理10对数据集
for i in range(1, 11):
    print(f"Processing group {i}...")

    # 文件路径
    train_file = f'train{i}.csv'  # 根据实际路径调整
    test_file = f'test{i}.csv'  # 根据实际路径调整

    # 加载数据
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # 提取特征和目标
    selected_features = ['PI-content', 'CF-content', 'GP-content', 'Sizing agent-content']  # 假设用三个特征
    X_train = train_data[selected_features]
    y_train = train_data.iloc[:, 0]
    X_test = test_data[selected_features]
    y_test = test_data.iloc[:, 0]

    # 初始化梯度提升树模型
    gbt = XGBRegressor(
        max_depth=3,
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    # 使用LOOCV替换原来的KFold
    loocv = LeaveOneOut()  # 修改2：使用LeaveOneOut
    train_r2_scores = []
    train_rmse_scores = []
    all_val_preds = []  # 修改3：收集所有验证集预测
    all_val_trues = []  # 修改3：收集所有验证集真实值

    for train_index, val_index in loocv.split(X_train):
        X_train_cv, X_val_cv = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]

        # 拟合模型
        gbt.fit(X_train_cv, y_train_cv)

        # 预测并计算指标
        y_pred_train_cv = gbt.predict(X_train_cv)
        y_pred_val_cv = gbt.predict(X_val_cv)

        # 收集验证集的预测和真实值
        all_val_preds.append(y_pred_val_cv[0])  # 修改4：提取标量值
        all_val_trues.append(y_val_cv.iloc[0])  # 修改4：提取标量值

        # 计算训练集指标
        train_rmse = np.sqrt(mean_squared_error(y_train_cv, y_pred_train_cv))
        train_r2 = r2_score(y_train_cv, y_pred_train_cv)
        train_rmse_scores.append(train_rmse)
        train_r2_scores.append(train_r2)

    # 计算验证集的整体指标
    test_rmse_cv = np.sqrt(mean_squared_error(all_val_trues, all_val_preds))  # 修改5：整体计算RMSE
    test_r2_cv = r2_score(all_val_trues, all_val_preds)  # 修改5：整体计算R²

    # 计算平均训练指标
    avg_train_rmse = np.mean(train_rmse_scores)
    avg_train_r2 = np.mean(train_r2_scores)

    # 使用整个训练集重新训练模型
    gbt.fit(X_train, y_train)

    # 提取规则
    booster_dump = gbt.get_booster().get_dump()

    # 在测试集上评估
    y_pred_test = gbt.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)

    # 保存结果
    results.append({
        "group": i,
        "train_rmse": avg_train_rmse,
        "test_rmse_cv": test_rmse_cv,  # 修改6：使用整体计算的CV指标
        "test_rmse_final": test_rmse,
        "train_r2": avg_train_r2,
        "test_r2_cv": test_r2_cv,  # 修改6：使用整体计算的CV指标
        "test_r2_final": test_r2,
        "rules": booster_dump
    })

# 转换为DataFrame以便分析
results_df = pd.DataFrame(results)

# 找到表现最优的一组
best_group = results_df.loc[results_df['test_rmse_final'].idxmin()]

# 打印最优结果
print("Best Group Results:")
print(f"Group: {best_group['group']}")
print(f"Train RMSE (Cross-Validation): {best_group['train_rmse']}")
print(f"Test RMSE (Cross-Validation): {best_group['test_rmse_cv']}")
print(f"Test RMSE (Final): {best_group['test_rmse_final']}")
print(f"Train R² (Cross-Validation): {best_group['train_r2']}")
print(f"Test R² (Cross-Validation): {best_group['test_r2_cv']}")
print(f"Test R² (Final): {best_group['test_r2_final']}")
print(f"Best Rules: {best_group['rules']}")

# 保存所有结果
results_df.to_csv("gradient_boosting_results.csv", index=False)
print("All results have been saved to 'gradient_boosting_results.csv'.")