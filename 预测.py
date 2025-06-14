import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import IsolationForest
import joblib
import os
import matplotlib.pyplot as plt

# 路径配置
DATA_PATH = r"F:\ai\ocr_recognition\dataset\正确的数据.csv"
MODEL_DIR = r"F:\ai\ocr_recognition\models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 异常检测模型保存路径
ANOMALY_MODEL_PATH = os.path.join(MODEL_DIR, "anomaly_detector.pkl")
# 分类趋势模型保存路径（目录）
CATEGORY_MODEL_DIR = os.path.join(MODEL_DIR, "category_prophet_models")
os.makedirs(CATEGORY_MODEL_DIR, exist_ok=True)

FORECAST_SAVE_PATH = r"F:\ai\ocr_recognition\dataset\forecast_next_30_days_by_category.csv"

def load_data():
    try:
        df = pd.read_csv(DATA_PATH, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_PATH, encoding='gbk')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    return df

def train_anomaly_detection_model(df):
    X = df[['total_amount']].values
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X)
    joblib.dump(model, ANOMALY_MODEL_PATH)
    print(f"异常检测模型已保存至 {ANOMALY_MODEL_PATH}")
    return model

def load_anomaly_detection_model():
    if os.path.exists(ANOMALY_MODEL_PATH):
        return joblib.load(ANOMALY_MODEL_PATH)
    else:
        print("异常检测模型不存在，请先训练！")
        return None

def train_or_load_prophet_model_by_category(df, category):
    model_path = os.path.join(CATEGORY_MODEL_DIR, f"prophet_{category}.pkl")
    if os.path.exists(model_path):
        print(f"加载分类 {category} 的模型...")
        model = joblib.load(model_path)
    else:
        print(f"训练分类 {category} 的消费趋势模型...")
        df_cat = df[df['category_cn'] == category]
        if len(df_cat) < 10:
            print(f"分类 {category} 数据太少，跳过训练")
            return None
        df_cat = df_cat[['date', 'total_amount']].rename(columns={'date': 'ds', 'total_amount': 'y'})
        model = Prophet()
        model.fit(df_cat)
        joblib.dump(model, model_path)
        print(f"模型已保存至 {model_path}")
    return model

def predict_next_30_days_by_category(df):
    categories = df['category_cn'].unique()
    anomaly_model = load_anomaly_detection_model()
    if anomaly_model is None:
        anomaly_model = train_anomaly_detection_model(df)

    results = []

    for cat in categories:
        model = train_or_load_prophet_model_by_category(df, cat)
        if model is None:
            continue

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        forecast_30 = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30).copy()
        forecast_30['category_cn'] = cat

        # 负值截断，确保预测结果非负
        forecast_30['yhat'] = forecast_30['yhat'].apply(lambda x: max(x, 0))
        forecast_30['yhat_lower'] = forecast_30['yhat_lower'].apply(lambda x: max(x, 0))
        forecast_30['yhat_upper'] = forecast_30['yhat_upper'].apply(lambda x: max(x, 0))

        # 异常检测针对预测值
        forecast_30['anomaly'] = anomaly_model.predict(forecast_30[['yhat']].values)
        forecast_30['anomaly'] = forecast_30['anomaly'].apply(lambda x: '异常' if x == -1 else '正常')

        results.append(forecast_30)

        print(f"\n分类: {cat} 未来30天消费预测和异常检测:")
        print(forecast_30)

        # 绘制预测趋势图
        fig = model.plot(forecast)
        plt.title(f"分类 {cat} 未来30天消费预测趋势")
        plt.show()

    if results:
        df_result = pd.concat(results)
        df_result.to_csv(FORECAST_SAVE_PATH, index=False, encoding='utf-8-sig')
        print(f"\n所有分类的预测结果已保存至 {FORECAST_SAVE_PATH}")

if __name__ == "__main__":
    df = load_data()
    predict_next_30_days_by_category(df)
