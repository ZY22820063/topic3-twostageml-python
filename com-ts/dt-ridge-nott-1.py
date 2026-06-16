import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, _tree
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


# =============== 1) 辅助函数：递归提取决策树叶子节点规则 ===============
def retrieve_leaf_rules(decision_tree, feature_names):
    """
    返回一个字典：
      leaf_id -> {
         "rule": 该叶节点的条件字符串,
         "samples": 该叶节点包含的样本数
      }
    """
    tree_ = decision_tree.tree_
    feat_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    leaf_dict = {}
    
    def recurse(node, path_conditions):
        # 如果不是叶子节点，继续拆分
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            feature = feat_name[node]
            threshold = tree_.threshold[node]
            # 左子树 (<= threshold)
            recurse(
                tree_.children_left[node], 
                path_conditions + [f"{feature} <= {threshold:.4f}"]
            )
            # 右子树 (> threshold)
            recurse(
                tree_.children_right[node], 
                path_conditions + [f"{feature} > {threshold:.4f}"]
            )
        else:
            # 叶子节点
            leaf_id = node
            samples = tree_.n_node_samples[node]
            if path_conditions:
                rule_str = "if " + " and ".join(path_conditions)
            else:
                rule_str = "if (no condition)"
            
            leaf_dict[leaf_id] = {
                "rule": rule_str,
                "samples": samples
            }
    
    # 从根节点开始
    recurse(0, [])
    return leaf_dict


# =============== 2) 主流程：训练决策树、提取规则、对每个叶子分段做多项式 Ridge ===============
# 加载整个数据集
data = pd.read_csv("composites.csv")  # 输入数据集文件
selected_features = ['PI-content', 'CF-content', 'GP-content', 'Sizing agent-content']  # 这里假设用这3个特征
X = data[selected_features]
y = data.iloc[:, 0]  # 假设目标变量是第一列

# 基于规则划分数据集
tree = DecisionTreeRegressor(max_depth=3, random_state=42)
tree.fit(X, y)

# 提取叶子节点信息（包含规则和样本数）
leaf_info_dict = retrieve_leaf_rules(tree, selected_features)

# 给全数据打上叶子编号
leaf_ids = tree.apply(X)
unique_leaf_ids = np.unique(leaf_ids)

results = []  # 用于保存每个叶子分段的结果

for leaf_id in unique_leaf_ids:
    # 该叶子节点的规则字符串（如“if temperature <= 405 and time > 17.5”）
    leaf_rule_str = leaf_info_dict[leaf_id]["rule"]
    samples_count = leaf_info_dict[leaf_id]["samples"]
    
    print(f"Processing leaf ID {leaf_id} with rule: {leaf_rule_str}")
    
    # 分段数据
    segment_mask = (leaf_ids == leaf_id)
    X_segment = X[segment_mask]
    y_segment = y[segment_mask]
    
    # 若分段数据过少，则跳过
    if X_segment.shape[0] < 4:
        print(f"Leaf ID {leaf_id} skipped due to insufficient data (<4).")
        continue
    
    # 自动划分训练集和测试集 (80%:20%)
    X_train_segment, X_test_segment, y_train_segment, y_test_segment = train_test_split(
        X_segment, y_segment, test_size=0.2, random_state=42
    )
    
    # 多项式特征
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_segment)
    X_test_poly = poly.transform(X_test_segment)
    
    # Ridge 回归
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train_poly, y_train_segment)
    
    # 提取公式
    intercept = ridge.intercept_
    coefs = ridge.coef_
    feat_names_poly = poly.get_feature_names_out(selected_features)
    
    # 为了避免 Excel 的 #NAME?，做一些替换：^2 -> _sq，空格->_x_
    safe_feat_names = []
    for f_name in feat_names_poly:
        f_name = f_name.replace("^2", "_sq").replace(" ", "_x_")
        safe_feat_names.append(f_name)
    
    # 组合成一个字符串公式
    terms = []
    for c, nm in zip(coefs, safe_feat_names):
        if abs(c) > 0.01:  # 过滤掉绝对值较小的系数，防止公式太臃肿
            terms.append(f"({c:.4f})*{nm}")
    if terms:
        formula_str = f"{intercept:.4f} + " + " + ".join(terms)
    else:
        formula_str = f"{intercept:.4f}"
    
    # 训练集评估
    y_train_pred = ridge.predict(X_train_poly)
    train_rmse = np.sqrt(mean_squared_error(y_train_segment, y_train_pred))
    train_r2 = r2_score(y_train_segment, y_train_pred)
    
    # 测试集评估
    y_test_pred = ridge.predict(X_test_poly)
    test_rmse = np.sqrt(mean_squared_error(y_test_segment, y_test_pred))
    test_r2 = r2_score(y_test_segment, y_test_pred)
    
    # 保存结果
    results.append({
        "leaf_id": leaf_id,
        "rule": leaf_rule_str,
        "train_size": X_train_segment.shape[0],
        "test_size": X_test_segment.shape[0],
        "piecewise_formula": formula_str,
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "test_rmse": test_rmse,
        "test_r2": test_r2
    })

# 整理结果
results_df = pd.DataFrame(results)
print("\n================= Final Results =================")
print(results_df)

# 找到测试集 RMSE 最小的一组
best_idx = results_df["test_rmse"].idxmin()
best_result = results_df.loc[best_idx]
print("\nBest Result:")
print(best_result)

# 保存所有结果
results_df.to_csv("auto_split_dt_ridge-1.csv", index=False, encoding="utf-8-sig")
print("\nAll results have been saved to 'auto_split_dt_ridge.csv'.")
