import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from gplearn.genetic import SymbolicRegressor
from sklearn.preprocessing import PolynomialFeatures

# 保存每个分段的结果
results = []

# 加载整个数据集
data = pd.read_csv("composites.csv")  # 根据实际情况修改文件名或路径
selected_features = ['PI-content', 'CF-content', 'GP-content', 'Sizing agent-content']
X = data[selected_features]
y = data.iloc[:, 0]  # 假设目标变量在第 0 列

# 训练决策树生成规则
tree = DecisionTreeRegressor(max_depth=3, random_state=42)
tree.fit(X, y)

# 导出决策树规则
rules = export_text(tree, feature_names=selected_features)
print("Decision Tree Rules:\n", rules)

# 获取每个样本所处的叶子节点
leaf_ids = tree.apply(X)
unique_leaf_ids = np.unique(leaf_ids)

# 按叶子节点分段，依次建模
for leaf_id in unique_leaf_ids:
    print(f"\nProcessing leaf ID {leaf_id}...")

    # 筛选当前叶子节点对应的数据
    segment_mask = (leaf_ids == leaf_id)
    X_segment = X[segment_mask]
    y_segment = y[segment_mask]

    # 如果分段数据过少，直接跳过
    if X_segment.shape[0] < 2:
        print(f"Leaf ID {leaf_id} skipped because samples={X_segment.shape[0]} < 2.")
        continue

    # 仅做简单的训练集-测试集划分
    X_train_seg, X_test_seg, y_train_seg, y_test_seg = train_test_split(
        X_segment, y_segment, test_size=0.2, random_state=42
    )

    # 如果需要多项式特征，则把 use_polynomial 改为 True
    use_polynomial = False
    if use_polynomial:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_seg = poly.fit_transform(X_train_seg)
        X_test_seg = poly.transform(X_test_seg)

    # 初始化符号回归模型
    regr = SymbolicRegressor(
        population_size=2000,
        generations=20,
        tournament_size=20,
        stopping_criteria=0.01,
        const_range=(-1.0, 1.0),
        function_set=['add', 'sub', 'mul', 'div'],
        metric='rmse',  # 若报错，可改成 'pearson' 等 gplearn 支持的 metric
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=1.0,
        parsimony_coefficient=0.0005,
        random_state=42,
        n_jobs=-1
    )

    # 训练符号回归模型
    regr.fit(X_train_seg, y_train_seg)

    # 在训练集上评估
    y_pred_train = regr.predict(X_train_seg)
    train_rmse = np.sqrt(mean_squared_error(y_train_seg, y_pred_train))
    train_r2 = r2_score(y_train_seg, y_pred_train)

    # 在测试集上评估
    if X_test_seg.shape[0] > 0:
        y_pred_test = regr.predict(X_test_seg)
        test_rmse = np.sqrt(mean_squared_error(y_test_seg, y_pred_test))
        # 若测试集只剩 1 条，同样会出现 R^2 警告，可用 try-except 处理或使用 np.nan
        if X_test_seg.shape[0] >= 2:
            test_r2 = r2_score(y_test_seg, y_pred_test)
        else:
            print(f"Warning: test set only has {X_test_seg.shape[0]} sample(s). R^2 is not well-defined.")
            test_r2 = np.nan
    else:
        # 如果当前分段没有留出测试数据，就记为 NaN
        test_rmse = np.nan
        test_r2 = np.nan

    # 获取符号回归表达式（内部属性，可能随版本调整）
    best_expr = str(regr._program)

    # 从 rules 文本中获取这一叶子节点对应的规则行（简易拼接示例）
    rule_lines = rules.splitlines()
    rule_str = (
        f"Leaf ID {leaf_id} rule: {rule_lines[leaf_id]}"
        if leaf_id < len(rule_lines) else f"Leaf ID {leaf_id}"
    )

    # 保存结果
    results.append({
        "leaf_id": leaf_id,
        "rule": rule_str,
        "train_size": X_train_seg.shape[0],
        "test_size": X_test_seg.shape[0],
        "symbolic_expression": best_expr,
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "test_rmse": test_rmse,
        "test_r2": test_r2
    })

# 将结果整理为 DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv("dt-symbolic_regressor_leafs.csv", index=False)

# 输出测试集 RMSE 最小的分段（若全是 NaN 则会报错，可自行加判断）
if not results_df["test_rmse"].isna().all():
    best_index = results_df["test_rmse"].idxmin()
    best_segment = results_df.loc[best_index]
    print("\nBest Segment Based on Test RMSE:")
    print(best_segment)
else:
    print("\nNo valid test_rmse found (all NaN).")
