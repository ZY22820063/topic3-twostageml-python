# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
import numpy as np

# 初始化结果列表
all_results = []
all_feature_subsets = []

# 循环处理10对数据集，这里假设有 train1.csv, test1.csv 至 train10.csv, test10.csv
for i in range(1, 11):
    train_file = f'train{i}.csv'  # 根据实际路径调整
    test_file = f'test{i}.csv'    # 根据实际路径调整
    
    # 加载训练集和测试集
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    X_train = train_data.iloc[:, 1:]  # 训练集特征
    y_train = train_data.iloc[:, 0]   # 训练集目标变量
    X_test = test_data.iloc[:, 1:]    # 测试集特征
    y_test = test_data.iloc[:, 0]     # 测试集目标变量
    
    # 特征数量范围（示例：从 4 个特征逐步减少到 1 个特征，可根据实际情况调整）
    feature_range = range(7, 0, -1)

    # 存储当前数据集的结果
    results = []
    feature_subsets = []

    for n_features in feature_range:
        print(f"数据集 {i} | 特征数 {n_features} | 开始")
        
        # 使用 ElasticNet 和 RFE 进行特征选择
        enet = ElasticNet(random_state=42)
        selector = RFE(estimator=enet, n_features_to_select=n_features, step=1)
        selector.fit(X_train, y_train)
        
        # 选择的特征
        selected_features = X_train.columns[selector.support_]
        feature_subsets.append({
            'Dataset': i,
            'n_features': n_features,
            'Selected_Features': selected_features.tolist()
        })
        
        # 使用选择的特征重新训练 ElasticNet 模型
        enet_model = ElasticNet(random_state=42)
        enet_model.fit(X_train[selected_features], y_train)
        
        # 在训练集上评估模型
        y_pred_train = enet_model.predict(X_train[selected_features])
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_r2 = r2_score(y_train, y_pred_train)
        
        # 在测试集上评估模型
        y_pred_test = enet_model.predict(X_test[selected_features])
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_r2 = r2_score(y_test, y_pred_test)
        
        # 使用 LOOCV 进行交叉验证
        loo = LeaveOneOut()
        cv_rmse_scores = -cross_val_score(
            enet_model,
            X_train[selected_features],
            y_train,
            scoring='neg_root_mean_squared_error',
            cv=loo,
            n_jobs=-1
        )
        cv_r2_scores = cross_val_score(
            enet_model,
            X_train[selected_features],
            y_train,
            scoring='r2',
            cv=loo,
            n_jobs=-1
        )
        cv_rmse = cv_rmse_scores.mean()
        cv_r2 = cv_r2_scores.mean()
        
        # 保存当前迭代的结果
        results.append({
            'Dataset': i,
            'n_features': n_features,
            'Train_RMSE': train_rmse,
            'Train_R2': train_r2,
            'CV_RMSE': cv_rmse,
            'CV_R2': cv_r2,
            'Test_RMSE': test_rmse,
            'Test_R2': test_r2
        })
        
        print(f"数据集 {i} | 特征数 {n_features} | 结束")
    
    # 将当前数据集的结果添加到总结果列表中
    all_results.extend(results)
    all_feature_subsets.extend(feature_subsets)

# 将所有结果转换为 DataFrame 并打印
all_results_df = pd.DataFrame(all_results)
print(all_results_df)

# 将所有特征子集转换为 DataFrame
all_feature_subsets_df = pd.DataFrame(all_feature_subsets)
print(all_feature_subsets_df)

# 将结果保存为 CSV 文件
all_results_df.to_csv('rfe_elasticnet_results.csv', index=False)
all_feature_subsets_df.to_csv('rfe_elasticnet_feature_subsets.csv', index=False)
