import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from gplearn.genetic import SymbolicRegressor

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
    selected_features = ['time', 'temperature', 'pressure', 'PI-content', 'CF-content', 'GP-content', 'Sizing agent-content']  # 假设用三个特征
    X_train = train_data[selected_features]
    y_train = train_data.iloc[:, 0]
    X_test = test_data[selected_features]
    y_test = test_data.iloc[:, 0]
    
    # 初始化神经网络回归模型
    nn_model = MLPRegressor(hidden_layer_sizes=(100, ), max_iter=500, random_state=42)
    nn_model.fit(X_train, y_train)
    
    # 神经网络预测输出
    nn_train_pred = nn_model.predict(X_train)
    nn_test_pred = nn_model.predict(X_test)
    
    # 将神经网络输出作为特征之一来训练符号回归
    X_train_with_nn = np.column_stack([X_train, nn_train_pred])  # 加入神经网络输出
    X_test_with_nn = np.column_stack([X_test, nn_test_pred])     # 加入神经网络输出

    # 初始化符号回归模型
    symbol_regressor = SymbolicRegressor(
        population_size=5000, generations=20, tournament_size=20, 
        stopping_criteria=0.01, const_range=(0, 1), init_depth=(2, 6),
        init_method='half and half', function_set=['add', 'sub', 'mul', 'div'],
        metric='mean absolute error', p_crossover=0.7, p_subtree_mutation=0.1,
        p_hoist_mutation=0.05, p_point_mutation=0.1, max_samples=1.0,
        random_state=42
    )
    
    # 10折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_r2_scores = []
    test_r2_scores = []
    train_rmse_scores = []
    test_rmse_scores = []
    
    for train_index, val_index in kf.split(X_train_with_nn):
        X_train_cv, X_val_cv = X_train_with_nn[train_index], X_train_with_nn[val_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]
        
        # 拟合符号回归模型
        symbol_regressor.fit(X_train_cv, y_train_cv)
        
        # 预测并计算指标
        y_pred_train_cv = symbol_regressor.predict(X_train_cv)
        y_pred_val_cv = symbol_regressor.predict(X_val_cv)
        train_rmse_scores.append(np.sqrt(mean_squared_error(y_train_cv, y_pred_train_cv)))
        test_rmse_scores.append(np.sqrt(mean_squared_error(y_val_cv, y_pred_val_cv)))
        train_r2_scores.append(r2_score(y_train_cv, y_pred_train_cv))
        test_r2_scores.append(r2_score(y_val_cv, y_pred_val_cv))
    
    # 计算平均指标
    avg_train_rmse = np.mean(train_rmse_scores)
    avg_test_rmse = np.mean(test_rmse_scores)
    avg_train_r2 = np.mean(train_r2_scores)
    avg_test_r2 = np.mean(test_r2_scores)
    
    # 使用整个训练集重新训练符号回归模型
    symbol_regressor.fit(X_train_with_nn, y_train)
    best_formula = symbol_regressor._program  # 获取最佳公式
    
    # 在测试集上评估
    y_pred_test = symbol_regressor.predict(X_test_with_nn)
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
        "formula": best_formula
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
print(f"Best Formula: {best_group['formula']}")

# 保存所有结果
results_df.to_csv("neural_network_symbolic-all.csv", index=False)
print("All results have been saved to 'neural_network_symbolic_regression_results.csv'.")
