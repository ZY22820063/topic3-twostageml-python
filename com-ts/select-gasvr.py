import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from tpot import TPOTRegressor
import numpy as np

# 初始化结果列表
all_results = []

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

    # 每次处理一个新的数据集时初始化 results 列表
    results = []

    # 多次运行遗传算法以获得多个结果集
    for run in range(5):
        # 将交叉验证方式从 KFold(n_splits=10) 改为 LeaveOneOut()
        tpot = TPOTRegressor(
            generations=5,
            population_size=20,
            verbosity=2,
            random_state=42,
            config_dict={
                'sklearn.svm.SVR': {
                    'kernel': ['linear', 'rbf'],
                    'C': [0.1, 1, 10],
                    'epsilon': [0.01, 0.1, 0.5]
                }
            },
            cv=LeaveOneOut()  # 使用 LOO 代替 KFold
        )

        # 训练TPOT模型
        tpot.fit(X_train, y_train)

        # 提取最佳管道并基于选择的特征训练SVM模型
        best_pipeline = tpot.fitted_pipeline_

        # 如果在最终管道中存在特征选择步骤，获取选出的特征
        selected_features = X_train.columns
        if 'feature_selection' in best_pipeline.named_steps:
            selected_features = X_train.columns[
                best_pipeline.named_steps['feature_selection'].get_support()
            ]

        # 在测试集上评估模型
        y_pred_test = best_pipeline.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_r2 = r2_score(y_test, y_pred_test)

        # LeaveOneOut 交叉验证
        cv_rmse = -cross_val_score(
            best_pipeline,
            X_train[selected_features],
            y_train,
            scoring='neg_root_mean_squared_error',
            cv=LeaveOneOut()
        ).mean()
        cv_r2 = cross_val_score(
            best_pipeline,
            X_train[selected_features],
            y_train,
            scoring='r2',
            cv=LeaveOneOut()
        ).mean()

        # 打印选出的特征数和特征名称
        print(f"Dataset {i}, Run {run + 1}: Selected {len(selected_features)} features.")
        print("Features:", selected_features.tolist())

        # 保存当前运行的结果
        results.append({
            'Dataset': i,
            'Run': run + 1,  # 增加一个运行编号
            'CV_RMSE': cv_rmse,
            'CV_R2': cv_r2,
            'Test_RMSE': test_rmse,
            'Test_R2': test_r2,
            'Num_Features': len(selected_features),
            'Selected_Features': ', '.join(selected_features)
        })

    # 将当前数据集的结果添加到总结果列表中
    all_results.extend(results)

# 将所有结果转换为DataFrame并打印
all_results_df = pd.DataFrame(all_results)
print(all_results_df)

# 可以选择将结果保存为CSV文件
all_results_df.to_csv('ga-svr-multiple.csv', index=False)
