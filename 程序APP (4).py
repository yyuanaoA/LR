#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 1. 加载模型
model = joblib.load('lr.pkl')

# 2. 加载训练集背景数据（用于SHAP解释）
X_train = pd.read_csv('X_train_background.csv')

# 3. 定义前10个特征及范围（确保名称和训练集一致，注意无多余空格）
top_10_features = [
    "Glycohemoglobin",
    "Glucose",
    "BRI",
    "TC",
    "BMI",
    "SII",
    "Hypertensiontime",
    "NHHR",
    "HDLC",
    "SIRI"
]

# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
    "Glycohemoglobin": {"type": "numerical", "min": 4.000, "max": 15.400, "default": 6.800},
    "Glucose": {"type": "numerical", "min": 47.000, "max": 554.000, "default": 111.000},
    "BRI": {"type": "numerical", "min": 2.756, "max": 18.297, "default": 4.177},
    "TC": {"type": "numerical", "min": 76, "max": 428.000, "default": 149},
    "BMI": {"type": "numerical", "min": 24, "max": 75.700, "default": 25.000},
    "SII": {"type": "numerical", "min": 41, "max": 3551.18, "default": 138.000},
    "Hypertensiontime": {"type": "numerical", "min": 0, "max": 63.000, "default": 13.000},
    "NHHR": {"type": "numerical", "min": 75, "max": 427, "default": 148.000},
    "HDLC": {"type": "numerical", "min": 5, "max": 122, "default": 54.000},
    "SIRI": {"type": "numerical", "min": 0.070, "max": 14.140, "default": 0.38},

}

# 2. Streamlit界面
st.title("Prediction Model with SHAP Visualization")

# 3. 动态输入框
feature_values = []
for feature in top_10_features:
    props = feature_ranges[feature]
    if props["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({props['min']} - {props['max']})",
            min_value=float(props["min"]),
            max_value=float(props["max"]),
            value=float(props["default"]),
        )
    else:
        value = st.selectbox(feature, props["options"])
    feature_values.append(value)

# 转为 DataFrame，列名必须与训练一致
features = pd.DataFrame([feature_values], columns=top_10_features)

# 4. 预测 + 展示结果
if st.button("Predict"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0][predicted_class] * 100

    st.write(f"Based on feature values, predicted possibility of Diabetes is {predicted_proba:.2f}%")

    # 5. 生成 SHAP waterfall 图
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(features)

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

