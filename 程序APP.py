#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载保存的模型
model = joblib.load('lr.pkl')

# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
    "Glycohemoglobin ": {"type": "numerical", "min": 4.000, "max": 15.400, "default": 6.800},
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

# Streamlit 界面
st.title("Prediction Model with SHAP Visualization")

# 动态生成输入
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

features = np.array([feature_values])

if st.button("Predict"):
    # 预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    probability = predicted_proba[predicted_class] * 100

    st.write(f"Based on feature values, predicted possibility of Diabetes is {probability:.2f}%")


    # 读取背景数据
    background = pd.read_csv("X_train_background.csv").sample(100, random_state=42)

    # 创建解释器
    explainer = shap.Explainer(model, background)
    shap_values = explainer(pd.DataFrame(features, columns=feature_ranges.keys()))

    # 画SHAP力图，直接显示
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.force(shap_values[0], matplotlib=True)
    st.pyplot(fig)

