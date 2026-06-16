import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import LeaveOneOut  # 修改导入
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 加载数据集
train_data = pd.read_csv('train9.csv')  # 替换为正确的文件路径
test_data = pd.read_csv('test9.csv')    # 替换为正确的文件路径

# 选定的特征
selected_features = ['PI-content', 'CF-content', 'Sizing agent-content', 'GP-content']

# 分离特征和目标
X_train = train_data[selected_features]
y_train = train_data.iloc[:, 0]
X_test = test_data[selected_features]
y_test = test_data.iloc[:, 0]

# 定义模型参数（修改为XGBoost参数）
params = {
    'learning_rate': 0.2,
    'max_depth': 7,
    'subsample': 0.6,
    'n_estimators': 100,
    'reg_lambda': 3,
    'objective': 'reg:squarederror',
    'random_state': 42,
    'n_jobs': -1,
    'colsample_bytree': 0.6
}

# 初始化XGBoost回归器
xgb = XGBRegressor(**params)

# ========== 修改开始：五折改为留一法 ==========
loo = LeaveOneOut()
train_r2_scores = []
test_r2_scores = []
train_rmse_scores = []
test_rmse_scores = []
all_val_preds = []
all_val_trues = []

for train_index, test_index in loo.split(X_train):
    X_train_cv, X_test_cv = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]

    xgb.fit(X_train_cv, y_train_cv)
    
    # 计算训练集指标
    y_pred_train_cv = xgb.predict(X_train_cv)
    train_rmse_scores.append(np.sqrt(mean_squared_error(y_train_cv, y_pred_train_cv)))
    train_r2_scores.append(r2_score(y_train_cv, y_pred_train_cv))
    
    # 收集验证集预测结果
    y_pred_test_cv = xgb.predict(X_test_cv)
    all_val_preds.append(y_pred_test_cv[0])
    all_val_trues.append(y_test_cv.values[0])

# 计算验证集整体指标
test_rmse = np.sqrt(mean_squared_error(all_val_trues, all_val_preds))
test_r2 = r2_score(all_val_trues, all_val_preds)
# ========== 修改结束 ==========

# 输出平均R2和RMSE分数
print(f"Average Train RMSE (LOO-CV): {np.mean(train_rmse_scores)}")
print(f"Validation RMSE (LOO-CV): {test_rmse}")
print(f"Average Train R2 (LOO-CV): {np.mean(train_r2_scores)}")
print(f"Validation R2 (LOO-CV): {test_r2}")

# 使用所有训练数据重新训练模型
xgb.fit(X_train, y_train)

# 对训练集和测试集进行预测
y_pred_train = xgb.predict(X_train)
y_pred_test = xgb.predict(X_test)

# 计算并打印训练集的最终 RMSE 和 R²
train_rmse_final = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_r2_final = r2_score(y_train, y_pred_train)
test_rmse_final = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2_final = r2_score(y_test, y_pred_test)

print(f"\nFinal Train RMSE: {train_rmse_final}")
print(f"Final Train R2: {train_r2_final}")
print(f"Final Test RMSE: {test_rmse_final}")
print(f"Final Test R2: {test_r2_final}")

# 保存预测结果
train_predictions = pd.DataFrame({'Actual': y_train, 'Predicted': y_pred_train})
test_predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_test})

#train_predictions.to_csv('train_predictions_xgb-tc-9-4.csv', index=False)
#test_predictions.to_csv('test_predictions_xgb-tc-9-4.csv', index=False)

print("\nTrain and test predictions have been saved.")

# 保存模型到文件（XGBoost专用格式）
#xgb.save_model("xgb_model-tc.model")  # 保存为二进制格式
xgb.save_model("xgb_model-tc.json")

print("XGBoost models saved in .model and .json formats.")