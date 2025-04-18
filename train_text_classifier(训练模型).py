from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import pickle
import os

# 读取数据
def load_data(file_path):
    df = pd.read_csv(file_path)
    # 处理 NaN 值，将 NaN 替换为空字符串
    df = df.fillna('')
    data = df['text']  # 这里 'text' 列是我们用来训练的文本数据
    labels = df['category']  # 'category' 列是目标分类标签
    return data, labels

# 训练分类器
def train_classifier(data, labels):
    # 使用 TfidfVectorizer 进行文本数据向量化
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(data)

    # 分割数据集为训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.2, random_state=42)

    # 使用 SMOTE 进行过采样来平衡数据
    smote = SMOTE(random_state=42, k_neighbors=1, sampling_strategy='auto')  # 调整邻居数和采样策略
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 使用 SVM 进行训练
    model = SVC(kernel='linear', C=1.0)
    model.fit(X_train_res, y_train_res)

    # 评估模型性能
    y_pred = model.predict(X_val)
    print(classification_report(y_val, y_pred, zero_division=1))

    # 保存模型和向量器
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    return model, vectorizer

# 模型加载函数
def load_model(model_path='best_model.pkl', vectorizer_path='vectorizer.pkl'):
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    else:
        print("没有找到已保存的模型，返回一个新的模型。")
        return None, None

# 主函数
def main():
    file_path = "F:/ai/ocr_recognition/dataset/train_labels_extracted.csv"
    data, labels = load_data(file_path)
    model, vectorizer = train_classifier(data, labels)

if __name__ == "__main__":
    main()
