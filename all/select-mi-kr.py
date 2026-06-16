# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import numpy as np

# 初始化结果列表
all_results = []
all_feature_subsets = []

# 循环处理10对数据集
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
    
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 计算互信息得分
    mi_scores = mutual_info_regression(X_train_scaled, y_train, random_state=42)
    mi_scores = pd.Series(mi_scores, index=X_train.columns)
    
    # 使用所有特征训练初始核岭回归模型
    kr_model_full = KernelRidge(kernel='rbf', alpha=1.0)
    kr_model_full.fit(X_train_scaled, y_train)
    
    # 计算排列重要性得分
    result = permutation_importance(kr_model_full, X_train_scaled, y_train,
                                    scoring='neg_mean_squared_error', n_repeats=10, random_state=42, n_jobs=-1)
    pi_scores = pd.Series(result.importances_mean, index=X_train.columns)
    
    # 标准化互信息得分和排列重要性得分
    mi_scores_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min())
    pi_scores_norm = (pi_scores - pi_scores.min()) / (pi_scores.max() - pi_scores.min())
    
    # 计算综合特征重要性得分
    combined_scores = mi_scores_norm + pi_scores_norm
    combined_scores = combined_scores.sort_values(ascending=False)
    
    # 特征数量范围（示例为从4递减到1）
    feature_range = range(7, 0, -1)

    # 存储当前数据集的结果
    results = []
    feature_subsets = []

    for n_features in feature_range:
        print(f"数据集 {i} | 特征数 {n_features} | 开始")
        
        # 选择综合得分最高的前 n_features 个特征
        selected_features = combined_scores.head(n_features).index.tolist()
        feature_subsets.append({
            'Dataset': i,
            'n_features': n_features,
            'Selected_Features': selected_features
        })
    
        # 使用 iloc 选择特征
        selected_indices = [X_train.columns.get_loc(feature) for feature in selected_features]
    
        # 基于选择的特征重新训练核岭回归模型
        kr_model = KernelRidge(kernel='rbf', alpha=1.0)
        kr_model.fit(X_train_scaled[:, selected_indices], y_train)
    
        # 在测试集上评估模型
        y_pred_test = kr_model.predict(X_test_scaled[:, selected_indices])
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_r2 = r2_score(y_test, y_pred_test)
    
        # 在训练集上评估模型
        y_pred_train = kr_model.predict(X_train_scaled[:, selected_indices])
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_r2 = r2_score(y_train, y_pred_train)
    
        # 留一交叉验证(LOO)
        loo = LeaveOneOut()
        cv_rmse_scores = -cross_val_score(kr_model,
                                          X_train_scaled[:, selected_indices],
                                          y_train,
                                          scoring='neg_root_mean_squared_error',
                                          cv=loo,
                                          n_jobs=-1)
        cv_r2_scores = cross_val_score(kr_model,
                                       X_train_scaled[:, selected_indices],
                                       y_train,
                                       scoring='r2',
                                       cv=loo,
                                       n_jobs=-1)
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
all_results_df.to_csv('kr_results.csv', index=False)
all_feature_subsets_df.to_csv('kr_feature_subsets.csv', index=False)
