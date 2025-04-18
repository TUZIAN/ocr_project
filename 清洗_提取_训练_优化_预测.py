import os
import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from paddleocr import PaddleOCR  # 用PaddleOCR进行OCR识别
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# OCR识别函数，自动根据图片内容选择语言模型
def ocr_recognition(image_path):
    ocr = PaddleOCR(use_angle_cls=True, lang='en')  # 默认使用英文模型
    result = ocr.ocr(image_path, cls=True)
    ocr_text = ''
    for line in result[0]:
        ocr_text += line[1][0] + ' '  # 修正：将元组中的文本部分提取出来
    if any(c in ocr_text for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"):  # 如果有英文字符
        ocr = PaddleOCR(use_angle_cls=True, lang='en')  # 如果有英文字符，则使用英文OCR
    else:
        ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # 否则使用中文OCR
        result = ocr.ocr(image_path, cls=True)
        ocr_text = ''
        for line in result[0]:
            ocr_text += line[1][0] + ' '  # 同样修正
    return ocr_text.strip()


# 数据清洗函数
def clean_text(text):
    """
    清理文本，去除特殊字符和多余的空格。
    """
    # 移除数字之外的字符
    text = re.sub(r'[^A-Za-z0-9\u4e00-\u9fa5]', ' ', text)  # 支持中英文
    # 去除多余空格
    text = ' '.join(text.split())
    return text

# 读取数据
train_labels_df = pd.read_csv('F:/ai/ocr_recognition/dataset/train_labels_extracted.csv')

# 获取字段数据
image_paths = train_labels_df['image_path'].tolist()  # 图片路径
texts = train_labels_df['text'].tolist()  # OCR文本
merchants = train_labels_df['merchant'].tolist()  # 商家
date_times = train_labels_df['date_time'].tolist()  # 日期时间
locations = train_labels_df['location'].tolist()  # 地点
items = train_labels_df['items'].tolist()  # 商品
total_amounts = train_labels_df['total_amount'].tolist()  # 总金额
categories = train_labels_df['category'].tolist()  # 类别

# OCR文本提取
ocr_texts = [ocr_recognition(image_path) for image_path in image_paths]

# 清洗后的文本
cleaned_texts = [clean_text(text) for text in ocr_texts]

# 特征提取（使用TF-IDF）
vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))  # 双词组
X = vectorizer.fit_transform(cleaned_texts)

# 标签
y = categories  # 使用类别作为标签

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM模型训练
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# 模型预测
y_pred = model.predict(X_test)

# 输出评估结果
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 可视化混淆矩阵
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# 模型保存
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 加载模型
with open('svm_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# 测试加载后的模型
new_image_paths = ["path/to/new/receipt.jpg"]  # 替换为新的OCR识别文本路径
new_ocr_texts = [ocr_recognition(image_path) for image_path in new_image_paths]
cleaned_new_texts = [clean_text(text) for text in new_ocr_texts]
X_new = vectorizer.transform(cleaned_new_texts)
predictions = loaded_model.predict(X_new)

print("Predicted category for new input:", predictions)

# 参数优化：使用网格搜索进行模型调参
param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
grid_search = GridSearchCV(SVC(kernel='linear'), param_grid, cv=5)

# 训练网格搜索模型
grid_search.fit(X_train, y_train)

# 输出最优参数和分数
print("Best parameters found by GridSearchCV:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# 数据增强（可选）：平衡类别
X_train_resampled, y_train_resampled = resample(X_train, y_train,
                                               replace=True,
                                               n_samples=len(y_train),
                                               random_state=42)

# 重新训练SVM模型
model_resampled = SVC(kernel='linear')
model_resampled.fit(X_train_resampled, y_train_resampled)

# 评估新模型
y_pred_resampled = model_resampled.predict(X_test)
print("Classification Report for Resampled Model:")
print(classification_report(y_test, y_pred_resampled))

# 可视化混淆矩阵
cm_resampled = confusion_matrix(y_test, y_pred_resampled)
sns.heatmap(cm_resampled, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# 新图片预测功能：选择小票图片并预测
def predict_new_receipt():
    # 使用图形界面选择文件
    Tk().withdraw()  # 不显示主窗口
    image_path = askopenfilename(title="选择小票图片", filetypes=[("Image files", "*.jpg *.jpeg *.png")])

    if image_path:
        # 识别OCR文本
        ocr_text = ocr_recognition(image_path)
        cleaned_text = clean_text(ocr_text)
        X_new = vectorizer.transform([cleaned_text])  # 特征提取
        prediction = loaded_model.predict(X_new)  # 预测

        print(f"预测结果：{prediction[0]}")  # 输出预测的消费类别
        return prediction[0]
    else:
        print("未选择图片。")

# 调用预测函数
predict_new_receipt()
