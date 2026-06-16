import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import RFE
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import time  # 导入时间模块

# ===== 新增：导入多项式特征生成器 =====
from sklearn.preprocessing import PolynomialFeatures

# 时间自适应显示函数（包括分钟、秒等详细格式）
def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{minutes} min {seconds:.2f} sec"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours} hr {minutes} min {seconds:.2f} sec"

# 初始化结果列表
all_results = []
all_feature_subsets = []

round = 11
# 循环处理20对数据集
for i in range(1, round):
    train_file = f'train{i}.csv'  # 根据实际路径调整
    test_file = f'test{i}.csv'    # 根据实际路径调整
    
    # 加载训练集和测试集
    try:
        train_data = pd.read_csv(train_file)
        test_data = pd.read_csv(test_file)
    except FileNotFoundError:
        print(f"文件未找到: {train_file} 或 {test_file}. 跳过数据集 {i}.")
        continue  # 跳过当前循环，继续下一个数据集
    
    # 分离特征和目标
    y_train = train_data.iloc[:, 0]         # 训练集目标变量
    X_train = train_data.iloc[:, 1:]        # 训练集特征
    y_test = test_data.iloc[:, 0]          # 测试集目标变量
    X_test = test_data.iloc[:, 1:]         # 测试集特征

    # ===== 新增：生成多项式特征（这里以二次多项式为例，可自行调整 degree） =====
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_train_poly_array = poly.fit_transform(X_train)
    X_test_poly_array = poly.transform(X_test)

    # 由于多项式转换后变为 numpy 数组，这里重新构造成 DataFrame，并且保留列名
    poly_feature_names = poly.get_feature_names_out(X_train.columns)
    X_train_poly = pd.DataFrame(X_train_poly_array, columns=poly_feature_names)
    X_test_poly = pd.DataFrame(X_test_poly_array, columns=poly_feature_names)

    # 确保数据集中至少有20个特征（此时已经是多项式扩展后的特征数）
    if X_train_poly.shape[1] < 14:
        print(f"数据集 {i} 的多项式特征数量少于20个。跳过特征选择。")
        continue  # 跳过当前循环，继续下一个数据集
    
    # 初始化CatBoost回归器，设置random_state以保证结果可重复
    catboost = CatBoostRegressor(verbose=0, random_state=42)
    
    # 特征数量范围（从 20 个特征减少到 5 个特征）
    feature_range = range(10, 0, -1)
    
    # 存储当前数据集的结果
    results = []
    feature_subsets = []
    
    # 初始化时间记录
    times_per_iteration = []

    for n_features in feature_range:
        print(f"i: {i} | n_features: {n_features} | start")

        # 记录开始时间
        start_time = time.time()
    
        # RFE进行特征选择
        selector = RFE(estimator=catboost, n_features_to_select=n_features, step=1)
        selector = selector.fit(X_train_poly, y_train)
    
        # 获取选择的特征
        selected_features = X_train_poly.columns[selector.support_]
    
        # 保存特征子集
        feature_subsets.append({
            'Dataset': i,
            'n_features': n_features,
            'Selected_Features': selected_features.tolist()
        })
    
        # 基于选择的特征重新训练CatBoost模型
        catboost_model = CatBoostRegressor(verbose=0, random_state=42)
        catboost_model.fit(X_train_poly[selected_features], y_train)
    
        # 在训练集上评估模型
        y_pred_train = catboost_model.predict(X_train_poly[selected_features])
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_r2 = r2_score(y_train, y_pred_train)
    
        # 十折交叉验证
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_rmse_scores = -cross_val_score(catboost_model, 
                                          X_train_poly[selected_features], 
                                          y_train, 
                                          scoring='neg_root_mean_squared_error', 
                                          cv=kf, 
                                          n_jobs=-1)
        cv_r2_scores = cross_val_score(catboost_model, 
                                       X_train_poly[selected_features], 
                                       y_train, 
                                       scoring='r2', 
                                       cv=kf, 
                                       n_jobs=-1)
        cv_rmse = cv_rmse_scores.mean()
        cv_r2 = cv_r2_scores.mean()
    
        # 在测试集上评估模型
        y_pred_test = catboost_model.predict(X_test_poly[selected_features])
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_r2 = r2_score(y_test, y_pred_test)
    
        # 记录结束时间
        end_time = time.time()
        iteration_time = end_time - start_time
        times_per_iteration.append(iteration_time)

        # 估算剩余时间
        avg_time_per_iteration = np.mean(times_per_iteration)
        # 剩余多少次迭代（包括当前数据集后续特征数量的迭代 + 后面数据集所有迭代）
        remaining_iterations = (len(feature_range) - feature_range.index(n_features) - 1) + (round - i - 1) * len(feature_range)
        estimated_time_remaining = avg_time_per_iteration * remaining_iterations

        # 自适应时间显示
        formatted_iteration_time = format_time(iteration_time)
        formatted_remaining_time = format_time(estimated_time_remaining)

        print(
            f"i: {i} | n_features: {n_features} | end | "
            f"iteration time: {formatted_iteration_time} | "
            f"estimated remaining time: {formatted_remaining_time}"
        )

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
    
    # 将当前数据集的结果添加到总结果列表中
    all_results.extend(results)
    all_feature_subsets.extend(feature_subsets)

# 将所有结果转换为 DataFrame 并打印
all_results_df = pd.DataFrame(all_results)
print(all_results_df)

# 将所有特征子集转换为 DataFrame 并打印
all_feature_subsets_df = pd.DataFrame(all_feature_subsets)
print(all_feature_subsets_df)

# 将结果保存为 CSV 文件
all_results_df.to_csv('poly_rfecatboost_all.csv', index=False)
all_feature_subsets_df.to_csv('poly_rfecatboost_feature_all.csv', index=False)

print("模型评估结果和特征子集已保存到CSV文件。")
