import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 初始化结果列表
all_results = []

# 循环处理10对数据集（1 到 10）
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
    
    # 初始化 Extra Trees 回归器
    extratrees = ExtraTreesRegressor(random_state=42)

    # 特征数量范围（这里仅作示例，你也可以使用你实际想要的范围）
    feature_range = range(7, 0, -1)

    # 存储当前数据集的结果
    results = []

    # LeaveOneOut（LOO）
    loo = LeaveOneOut()

    for n_features in feature_range:
        print(f"i: {i} | n_features: {n_features} | start")

        # RFE 进行特征选择
        selector = RFE(estimator=extratrees, n_features_to_select=n_features, step=1)
        selector = selector.fit(X_train, y_train)

        # 获取选择的特征
        selected_features = X_train.columns[selector.support_]

        # 基于选择的特征重新训练 Extra Trees 模型
        extratrees_model = ExtraTreesRegressor(random_state=42)
        extratrees_model.fit(X_train[selected_features], y_train)

        # ------------------
        # 1. 在训练集上进行预测并计算指标
        # ------------------
        train_pred = extratrees_model.predict(X_train[selected_features])
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_r2 = r2_score(y_train, train_pred)

        # 在测试集上评估模型
        y_pred_test = extratrees_model.predict(X_test[selected_features])
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_r2 = r2_score(y_test, y_pred_test)

        # 留一法交叉验证
        cv_rmse = -cross_val_score(
            extratrees_model, 
            X_train[selected_features], 
            y_train, 
            scoring='neg_root_mean_squared_error', 
            cv=loo
        ).mean()
        cv_r2 = cross_val_score(
            extratrees_model, 
            X_train[selected_features], 
            y_train, 
            scoring='r2', 
            cv=loo
        ).mean()

        # ------------------
        # 2. 保存当前迭代的结果（增加训练指标）
        # ------------------
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
        print(f"i: {i} | n_features: {n_features} | end")

    # 将当前数据集的结果添加到总结果列表中
    all_results.extend(results)

# 将所有结果转换为 DataFrame 并打印
all_results_df = pd.DataFrame(all_results)
print(all_results_df)

# 可以选择将结果保存为 CSV 文件
all_results_df.to_csv('rfeextratrees.csv', index=False)
