# ======================= 逆向设计示例代码（添加填料约束版） =======================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skopt import gp_minimize
from skopt.space import Real
from skopt.plots import plot_convergence, plot_objective
from xgboost import XGBRegressor

# -------------------------- 可视化设置 --------------------------
sns.set_theme(style="whitegrid", palette="pastel", font="Times New Roman")
plt.rcParams.update({
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 300
})

# -------------------------- 模型加载 --------------------------
xgb = XGBRegressor()
xgb.load_model("xgb_model.model")

# -------------------------- 参数空间定义（添加双重约束） --------------------------
space = [
    Real(0.5, 1.0, name='PI-content'),  # 基体材料占比50%-100%
    Real(0.0, 0.5, name='CF-content'),  # 增强纤维0%-50%
    Real(0.0, 0.1, name='Sizing agent-content'),  # 上浆剂0%-10%
    Real(0.01, 0.5, name='GP-content')  # 关键修改：填料必须≥1%
]

# -------------------------- 优化目标定义 --------------------------
TARGET = 70.0


def objective(params):
    pi, cf, sizing, gp = params
    total = sum(params)

    # 构建特征数据（必须首先执行）
    X = pd.DataFrame([params], columns=[
        'PI-content',
        'CF-content',
        'Sizing agent-content',
        'GP-content'
    ])
    pred = xgb.predict(X)[0]  # 关键：先获取预测值

    penalty = 0.0

    # 约束1：成分总和≤1.0
    if total > 1.0:
        penalty += 1e10 * (total - 1.0) ** 2

    # 约束2：当填料存在时（GP≥1%），上浆剂≤5%
    if gp > 0.01 and sizing > 0.05:
        penalty += 1e8 * (sizing - 0.05) ** 2

    return (pred - TARGET) ** 2 + penalty

    # 构建特征数据
    X = pd.DataFrame([params], columns=[
        'PI-content',
        'CF-content',
        'Sizing agent-content',
        'GP-content'
    ])

    pred = xgb.predict(X)[0]
    return (pred - TARGET) ** 2


# -------------------------- 执行优化 --------------------------
res = gp_minimize(
    objective, space,
    n_calls=500,  # 增加总迭代次数
    n_random_starts=100,  # 增加初始随机采样
    random_state=42,
    verbose=True
)

# -------------------------- 结果分析 --------------------------
best_params = res.x
best_pred = xgb.predict(pd.DataFrame([best_params], columns=[
    'PI-content',
    'CF-content',
    'Sizing agent-content',
    'GP-content'
]))[0]

print(f"\n【优化结果】")
print(f"最佳参数组合: {np.round(best_params, 4)}")
print(f"成分总和: {sum(best_params):.4f}")
print(f"预测值: {best_pred:.2f} (目标: {TARGET})")
print(f"绝对误差: {abs(best_pred - TARGET):.2f}")

# 其余可视化代码保持不变...

# -------------------------- 可视化分析 --------------------------
# 创建分析画布
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# 1. 收敛曲线
plot_convergence(res, ax=axs[0, 0])
axs[0, 0].set_title("优化收敛过程", fontsize=12)
axs[0, 0].grid(True, alpha=0.3)

# 2. 参数重要性
plot_objective(res, ax=axs[0, 1])
axs[0, 1].set_title("参数重要性分析", fontsize=12)
axs[0, 1].grid(True, alpha=0.3)

# 3. 目标值分布
axs[1, 0].hist(res.func_vals, bins=20, color='steelblue', edgecolor='white')
axs[1, 0].axvline(TARGET, color='red', linestyle='--', label='目标值')
axs[1, 0].set_title("目标值分布", fontsize=12)
axs[1, 0].set_xlabel("预测值")
axs[1, 0].set_ylabel("频次")
axs[1, 0].legend()
axs[1, 0].grid(True, alpha=0.3)

# 4. 参数分布矩阵
from skopt.plots import plot_evaluations

plot_evaluations(res, bins=10, ax=axs[1, 1])
axs[1, 1].set_title("参数分布矩阵", fontsize=12)
axs[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('优化分析.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------- 运行前准备 --------------------------
"""
请按以下步骤操作：
1. 安装必要库（在终端执行）：
   pip install seaborn scikit-optimize xgboost pandas matplotlib numpy

2. 确保以下文件存在：
   - xgb_model.model（XGBoost模型文件）
   - 数据文件（代码中已通过pandas加载）

3. 如果使用Anaconda，请先激活环境：
   conda activate your_env_name
"""