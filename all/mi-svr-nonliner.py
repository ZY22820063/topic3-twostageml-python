# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression
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

    # 初始化SVM模型，使用非线性核函数（如'rbf'）
    svm_model_full = SVR(kernel='rbf')

    # 使用所有特征训练初始SVM模型
    svm_model_full.fit(X_train, y_train)

    # 计算互信息得分
    mi_scores = mutual_info_regression(X_train, y_train)

    # 创建一个Series来保存互信息得分
    mi_scores_series = pd.Series(mi_scores, index=X_train.columns)

    # 标准化互信息得分
    mi_scores_norm = (mi_scores_series - mi_scores_series.min()) / (mi_scores_series.max() - mi_scores_series.min())

    # 特征数量范围（此处示例为 4 到 1，可根据实际需要调整）
    feature_range = range(7, 0, -1)

    # 存储当前数据集的结果
    results = []
    feature_subsets = []

    # 初始化留一法
    loo = LeaveOneOut()

    for n_features in feature_range:
        print(f"数据集 {i} | 特征数 {n_features} | 开始")

        # 选择互信息得分最高的前 n_features 个特征
        selected_features = mi_scores_norm.sort_values(ascending=False).head(n_features).index.tolist()
        feature_subsets.append({
            'Dataset': i,
            'n_features': n_features,
            'Selected_Features': selected_features
        })

        # 基于选择的特征重新训练SVM模型
        svm_model = SVR(kernel='rbf')
        svm_model.fit(X_train[selected_features], y_train)

        # 在测试集上评估模型
        y_pred_test = svm_model.predict(X_test[selected_features])
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_r2 = r2_score(y_test, y_pred_test)

        # 在训练集上评估模型
        y_pred_train = svm_model.predict(X_train[selected_features])
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_r2 = r2_score(y_train, y_pred_train)

        # 留一法交叉验证
        cv_rmse_scores = -cross_val_score(
            svm_model,
            X_train[selected_features],
            y_train,
            scoring='neg_root_mean_squared_error',
            cv=loo,
            n_jobs=-1
        )
        cv_r2_scores = cross_val_score(
            svm_model,
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
all_results_df.to_csv('ts_svm_mi_results.csv', index=False)
all_feature_subsets_df.to_csv('ts_svm_mi_feature_subsets.csv', index=False)
