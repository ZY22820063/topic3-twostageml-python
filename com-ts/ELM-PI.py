# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:39:57 2024

@author: 15297
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
from pyrcn.extreme_learning_machine import ELMRegressor  # 使用 pyrcn 库中的 ELM
from sklearn.preprocessing import PolynomialFeatures

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
    
    # 多项式特征扩展（这一步实现了非线性变换）
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # 初始化 ELM 模型
    elm = ELMRegressor(hidden_layer_size=50)  # 可以根据需要调整 hidden_layer_size
    
    # Leave-One-Out 交叉验证
    loo = LeaveOneOut()
    train_r2_scores = []
    test_r2_scores = []
    train_rmse_scores = []
    test_rmse_scores = []
    
    # 逐个样本进行 LOO
    for train_index, val_index in loo.split(X_train):
        X_train_cv, X_val_cv = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]
        
        # 对交叉验证数据进行多项式扩展
        X_train_cv_poly = poly.transform(X_train_cv)
        X_val_cv_poly = poly.transform(X_val_cv)
        
        # 拟合模型
        elm.fit(X_train_cv_poly, y_train_cv)
        
        # 预测并计算指标
        y_pred_train_cv = elm.predict(X_train_cv_poly)
        y_pred_val_cv = elm.predict(X_val_cv_poly)
        train_rmse_scores.append(np.sqrt(mean_squared_error(y_train_cv, y_pred_train_cv)))
        test_rmse_scores.append(np.sqrt(mean_squared_error(y_val_cv, y_pred_val_cv)))
        train_r2_scores.append(r2_score(y_train_cv, y_pred_train_cv))
        test_r2_scores.append(r2_score(y_val_cv, y_pred_val_cv))
    
    # 计算平均指标
    avg_train_rmse = np.mean(train_rmse_scores)
    avg_test_rmse = np.mean(test_rmse_scores)
    avg_train_r2 = np.mean(train_r2_scores)
    avg_test_r2 = np.mean(test_r2_scores)
    
    # 使用整个训练集重新训练模型
    elm.fit(X_train_poly, y_train)
    
    # 在测试集上评估
    y_pred_test = elm.predict(X_test_poly)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)
    
    # 保存结果
    results.append({
        "group": i,
        "train_rmse": avg_train_rmse,
        "test_rmse_cv": avg_test_rmse,
        "test_rmse_final": test_rmse,
        "train_r2": avg_train_r2,
        "test_r2_cv": avg_test_r2,
        "test_r2_final": test_r2,
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

# 保存所有结果
results_df.to_csv("elm_results.csv", index=False)
print("All results have been saved to 'elm_results.csv'.")
