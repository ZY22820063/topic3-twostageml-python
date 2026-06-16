# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 21:20:57 2025

@author: 15297
"""

import streamlit as st
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from skopt import gp_minimize
from skopt.space import Real

# Page title and description
st.title("Inverse Design Online Platform")
st.markdown("Optimize parameter combinations using CatBoost model and Bayesian optimization.")

# Load model
model = CatBoostRegressor()
model.load_model("cat_model5.cbm")

# Target performance value input
TARGET_VALUE = st.number_input("Enter target performance value (e.g., TS)", min_value=0.0, max_value=150.0, value=80.0)

# Dynamically define the search space based on target value
if 50 < TARGET_VALUE < 90:
    space = [
        Real(10, 90, name='time'),
        Real(400, 430, name='temperature'),
        Real(10, 20, name='pressure')
    ]
elif 90 <= TARGET_VALUE <= 110:
    space = [
        Real(35, 45, name='time'),
        Real(415, 425, name='temperature'),
        Real(15, 20, name='pressure')
    ]
else:
    st.warning("Target value is outside the supported optimization range (50–110). Please adjust the input.")
    st.stop()

# Objective function
def objective(params):
    time_val, temp_val, press_val = params
    X_features = pd.DataFrame({
        'time': [time_val],
        'temperature': [temp_val],
        'time^2': [time_val**2],
        'time temperature': [time_val * temp_val],
        'temperature^2': [temp_val**2],
        'temperature pressure': [temp_val * press_val],
    })
    pred = model.predict(X_features)[0]
    return (pred - TARGET_VALUE)**2

# Optimization trigger
if st.button("Start Inverse Design Optimization"):
    with st.spinner("Running optimization..."):
        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=100,
            n_random_starts=20,
            random_state=42
        )

    # Retrieve and display best results
    best_time, best_temp, best_press = result.x
    best_features = pd.DataFrame({
        'time': [best_time],
        'temperature': [best_temp],
        'time^2': [best_time**2],
        'time temperature': [best_time * best_temp],
        'temperature^2': [best_temp**2],
        'temperature pressure': [best_temp * best_press],
    })
    best_pred = model.predict(best_features)[0]

    st.success("Optimization completed! Optimal parameters:")
    st.write(f"**Time**: {best_time:.2f} min")
    st.write(f"**Temperature**: {best_temp:.2f} °C")
    st.write(f"**Pressure**: {best_press:.2f} MPa")
    st.write(f"**Predicted value**: {best_pred:.2f} — Target: {TARGET_VALUE}, Error: {abs(best_pred - TARGET_VALUE):.2f}")


'--------------------------------------------------------------------------------------------------'
import xgboost as xgb

st.markdown("---")
st.header("📊 Composite Property Prediction")

st.markdown("Input material composition ratios. The sum must be **1.0**.")

# 用户输入滑块
pi_content = st.slider("PI-content", 0.70, 1.00, 0.80, step=0.01)
cf_content = st.slider("CF-content", 0.0, 0.2, 0.1, step=0.005)
gp_content = st.slider("GP-content", 0.0, 0.2, 0.05, step=0.005)
sizing_content = st.slider("Sizing agent-content", 0.0, 0.1, 0.05, step=0.005)

# 总和校验
total = pi_content + cf_content + gp_content + sizing_content
if abs(total - 1.0) > 1e-6:
    st.error(f"❌ The total content must be 1.0. Current total: {total:.3f}")
else:
    if st.button("Predict TC & TS"):
        # 加载模型
        tc_model = xgb.Booster()
        ts_model = xgb.Booster()
        tc_model.load_model("xgb_model-tc-ratio.model")
        ts_model.load_model("xgb_model-ts-ratio.model")

        # 构造输入 DataFrame
        input_df = pd.DataFrame([{
            'PI-content': pi_content,
            'CF-content': cf_content,
            'GP-content': gp_content,
            'Sizing agent-content': sizing_content
        }])
        
        # 新增特征 CF/S
        input_df['CF/S'] = cf_content / sizing_content if sizing_content != 0 else 0.0

        # 转为 DMatrix 并预测
        dmatrix = xgb.DMatrix(input_df)
        tc_pred = tc_model.predict(dmatrix)[0]
        ts_pred = ts_model.predict(dmatrix)[0]

        # 输出结果
        st.success("✅ Prediction complete")
        st.write(f"**Thermal Conductivity (TC)**: `{tc_pred:.3f} W/m·K`")
        st.write(f"**Tensile Strength (TS)**: `{ts_pred:.2f} MPa`")

        if tc_pred >= 12 and ts_pred >= 70:
            st.markdown("🎯 **This is a high-performance material candidate!**")



