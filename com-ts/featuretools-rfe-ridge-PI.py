# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import featuretools as ft
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

# 保存每组结果
results = []

# 使用 Featuretools 自动生成特征的函数
def generate_features(train_df, test_df):
    # 添加索引
    train_df = train_df.reset_index(drop=True)
    train_df["index"] = range(len(train_df))
    test_df = test_df.reset_index(drop=True)
    test_df["index"] = range(len(test_df))

    # 创建 EntitySet
    es = ft.EntitySet(id="data")
    es.add_dataframe(dataframe_name="train", dataframe=train_df, index="index")

    # 自动特征工程 - 生成特征矩阵
    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="train",
        trans_primitives=["add_numeric", "multiply_numeric", "square_root", "absolute"],
        verbose=True
    )

    # 确保测试集生成相同的特征
    es_test = ft.EntitySet(id="data_test")
    es_test.add_dataframe(dataframe_name="test", dataframe=test_df, index="index")

    test_feature_matrix, _ = ft.dfs(
        entityset=es_test,
        target_dataframe_name="test",
        trans_primitives=["add_numeric", "multiply_numeric", "square_root", "absolute"],
        verbose=True
    )

    return feature_matrix, test_feature_matrix

# 循环处理10对数据集
for i in range(1, 11):
    print(f"Processing group {i}...")

    # 文件路径
    train_file = f'train{i}.csv'
    test_file = f'test{i}.csv'
    
    # 加载数据
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # 提取目标列和特征
    target_col = train_data.columns[0]
    y_train = train_data[target_col]
    y_test = test_data[target_col]

    # 生成自动特征
    X_train, X_test = generate_features(train_data.drop(columns=[target_col]), 
                                        test_data.drop(columns=[target_col]))

    # 标准化数据
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    feature_names = X_train.columns  # 保留特征名称

    # 初始化 Ridge 回归模型
    ridge = Ridge(alpha=1.0, random_state=42)

    # RFE 特征选择
    for num_features in range(min(X_train_scaled.shape[1], 10), 0, -1):
        print(f"Selecting top {num_features} features...")
        rfe = RFE(estimator=ridge, n_features_to_select=num_features, step=1)
        rfe.fit(X_train_scaled, y_train)

        # 筛选后的特征
        selected_features_mask = rfe.support_
        X_train_selected = X_train_scaled[:, selected_features_mask]
        X_test_selected = X_test_scaled[:, selected_features_mask]
        selected_feature_names = feature_names[selected_features_mask]

        # 模型训练
        ridge.fit(X_train_selected, y_train)

        # 构建公式
        coefficients = ridge.coef_
        intercept = ridge.intercept_
        formula = f"{intercept:.4f}" + "".join(
            f" + ({coef:.4f})*{feature}" for coef, feature in zip(coefficients, selected_feature_names)
        )

        # ========== 将 KFold 改为 LeaveOneOut ==========
        loo = LeaveOneOut()
        train_r2_scores = []
        test_r2_scores = []
        train_rmse_scores = []
        test_rmse_scores = []

        # 交叉验证评估（Leave-One-Out）
        for train_index, val_index in loo.split(X_train_selected):
            X_train_cv, X_val_cv = X_train_selected[train_index], X_train_selected[val_index]
            y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]

            ridge.fit(X_train_cv, y_train_cv)
            y_pred_train_cv = ridge.predict(X_train_cv)
            y_pred_val_cv = ridge.predict(X_val_cv)

            train_rmse_scores.append(np.sqrt(mean_squared_error(y_train_cv, y_pred_train_cv)))
            test_rmse_scores.append(np.sqrt(mean_squared_error(y_val_cv, y_pred_val_cv)))
            train_r2_scores.append(r2_score(y_train_cv, y_pred_train_cv))
            test_r2_scores.append(r2_score(y_val_cv, y_pred_val_cv))

        avg_train_rmse = np.mean(train_rmse_scores)
        avg_test_rmse = np.mean(test_rmse_scores)
        avg_train_r2 = np.mean(train_r2_scores)
        avg_test_r2 = np.mean(test_r2_scores)

        # 在测试集上评估最终模型
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

# 保存结果
results_df = pd.DataFrame(results)
results_df.to_csv("ridge_featuretools.csv", index=False)
print("All results have been saved to 'ridge_results_with_featuretools.csv'.")
