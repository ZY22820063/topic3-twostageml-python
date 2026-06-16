import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
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
    
    # 特征数量范围（从4递减到1）
    feature_range = range(7, 0, -1)

    # 存储当前数据集的结果
    results = []

    for n_features in feature_range:
        print(f"Processing dataset {i} | Features: {n_features}")
        
        # RFE进行特征选择
        selector = RFE(Ridge(), n_features_to_select=n_features, step=1)
        selector.fit(X_train, y_train)

        # 获取选择的特征
        selected_features = X_train.columns[selector.support_]

        # 训练最终模型
        ridge_model = Ridge()
        ridge_model.fit(X_train[selected_features], y_train)

        # === 新增部分：训练集评估 ===
        y_pred_train = ridge_model.predict(X_train[selected_features])
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_r2 = r2_score(y_train, y_pred_train)

        # 交叉验证评估
        loo = LeaveOneOut()
        cv_rmse = -cross_val_score(ridge_model, 
                                  X_train[selected_features], 
                                  y_train, 
                                  scoring='neg_root_mean_squared_error', 
                                  cv=loo).mean()
        cv_r2 = cross_val_score(ridge_model, 
                               X_train[selected_features], 
                               y_train, 
                               scoring='r2', 
                               cv=loo).mean()

        # 测试集评估
        y_pred_test = ridge_model.predict(X_test[selected_features])
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_r2 = r2_score(y_test, y_pred_test)

        # 保存结果
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
        
        print(f"Completed dataset {i} | Features: {n_features}")

    # 将当前数据集的结果添加到总结果列表中
    all_results.extend(results)

# 转换为DataFrame并保存
all_results_df = pd.DataFrame(all_results)
all_results_df.to_csv('rferidge_results.csv', index=False)

# 打印结果
print("\nFinal Results:")
print(all_results_df)