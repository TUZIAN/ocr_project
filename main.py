import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib
from paddleocr import PaddleOCR
import cv2
import tkinter as tk
from tkinter import filedialog

# 初始化OCR模型，中文和英文分别调用
ocr_ch = PaddleOCR(use_angle_cls=True, lang='ch')
ocr_en = PaddleOCR(use_angle_cls=True, lang='en')

def load_data(csv_path):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df = df[['text', 'category_cn']].dropna()
    return df

def vectorize_text(texts, max_features=3000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def train_model(X, y):
    clf = LinearSVC()
    clf.fit(X, y)
    return clf

def save_model(model, vectorizer, prefix):
    joblib.dump(model, prefix + '_svm.pkl')
    joblib.dump(vectorizer, prefix + '_vectorizer.pkl')
    print(f"模型已保存至 {prefix}_svm.pkl 和 {prefix}_vectorizer.pkl")

def load_model(prefix):
    model = joblib.load(prefix + '_svm.pkl')
    vectorizer = joblib.load(prefix + '_vectorizer.pkl')
    return model, vectorizer

def extract_text_lines_from_result(result):
    """
    修正后的提取函数：支持 PaddleOCR 输出格式
    result 是 PaddleOCR 返回的嵌套结构 [[[box, (text, score)], ...]]
    返回文本行列表
    """
    lines = []
    if not isinstance(result, list) or len(result) == 0 or not isinstance(result[0], list):
        print("OCR结果结构不符合预期")
        return lines
    for line in result[0]:  # 处理第一张图片
        if isinstance(line, list) and len(line) >= 2:
            text_info = line[1]
            if isinstance(text_info, (tuple, list)) and len(text_info) > 0:
                text = text_info[0]
                if isinstance(text, str):
                    lines.append(text.strip())
    return lines

def ocr_extract_text_multilang(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    print("CR文字提取中...")

    result_ch = ocr_ch.ocr(img, cls=True)
    print("OCR中文识别结果示例:", result_ch[0][:2])
    result_en = ocr_en.ocr(img, cls=True)
    print("OCR英文识别结果示例:", result_en[0][:2])

    lines_ch = extract_text_lines_from_result(result_ch)
    lines_en = extract_text_lines_from_result(result_en)

    text_ch = "\n".join(lines_ch)
    text_en = "\n".join(lines_en)
    final_text = text_ch if len(text_ch) > len(text_en) else text_en

    print(f"OCR全文本:\n{final_text}\n")
    return final_text

def choose_image_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="请选择测试图片",
        filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("所有文件", "*.*")]
    )
    return file_path

def extract_date(text):
    date_patterns = [
        r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
        r'(\d{1,2}[\'/-]\d{1,2}[\'/-]\d{2,4})',
        r'(\d{4}\.\d{1,2}\.\d{1,2})',
    ]
    for line in text.splitlines():
        for pattern in date_patterns:
            match = re.search(pattern, line)
            if match:
                date_str = match.group(1).replace("'", "/").replace("-", "/").replace(".", "/")
                return date_str
    return ""

def extract_amount(text):
    amounts = []
    for line in text.splitlines():
        found = re.findall(r'[\$]?(\d+\.\d{1,2})', line)
        if found:
            amounts.extend([float(a) for a in found])
    if amounts:
        return max(amounts)
    else:
        return None

def extract_merchant(text):
    lines = text.strip().split('\n')
    for line in lines[:5]:
        merchant = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9\s\-&]{3,30}', line)
        if merchant:
            candidate = merchant[0].strip()
            if not re.match(r'^\d+(\.\d+)?$', candidate) and candidate.lower() not in ['amount', 'total', 'description', 'items sold']:
                return candidate
    return ""

def train_and_save(csv_path, model_prefix):
    print("加载训练数据...")
    df = load_data(csv_path)
    print(f"训练样本数量: {len(df)}")
    X, vectorizer = vectorize_text(df['text'])
    y = df['category_cn']
    clf = train_model(X, y)
    save_model(clf, vectorizer, model_prefix)

def predict_category(text, model, vectorizer):
    X = vectorizer.transform([text])
    pred = model.predict(X)
    return pred[0]

def main():
    img_path = choose_image_file()
    if not img_path:
        print("未选择图片，程序退出")
        return

    full_text = ocr_extract_text_multilang(img_path)

    print("加载模型...")
    model, vectorizer = load_model('F:/ai/ocr_recognition/models/receipt_classifier_new')

    print("进行分类预测...")
    pred_category = predict_category(full_text, model, vectorizer)
    print(f"预测类别（中文）：{pred_category}")

    date = extract_date(full_text)
    amount = extract_amount(full_text)
    merchant = extract_merchant(full_text)

    print(f"提取日期: {date}")
    print(f"提取金额: {amount}")
    print(f"提取商户: {merchant}")

    result_df = pd.DataFrame([{
        'image_path': img_path,
        'text': full_text,
        'category': pred_category,
        'date': date,
        'amount': amount,
        'merchant': merchant
    }])
    save_path = 'F:/ai/ocr_recognition/dataset/ocr_predict_results.csv'
    if os.path.exists(save_path):
        result_df.to_csv(save_path, mode='a', header=False, index=False, encoding='utf-8-sig')
    else:
        result_df.to_csv(save_path, index=False, encoding='utf-8-sig')

    print(f"结果已保存到 {save_path}")

if __name__ == "__main__":
    main()
