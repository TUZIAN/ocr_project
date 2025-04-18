import re
import pandas as pd
import joblib
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

# 加载已训练模型和向量器
model = joblib.load("F:/ai/ocr_recognition/best_model.pkl")
vectorizer = joblib.load("F:/ai/ocr_recognition/vectorizer.pkl")

def extract_receipt_info(text):
    try:
        lines = text.strip().split('\n')
        merchant = lines[0] if lines else ""

        # 日期时间（中英文）
        date_time_match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}[ ,]*\d{1,2}:\d{2}(?:AM|PM)?)', text)
        date_time = date_time_match.group(1) if date_time_match else ""

        # 总金额（TOTAL / 总计）
        total_match = re.search(r'(TOTAL|总计|总额)[:： ]*\$?￥?(\d+\.\d{2})', text, re.IGNORECASE)
        total_amount = float(total_match.group(2)) if total_match else None

        # 地点
        location_match = re.search(r'(\d+ .*?(Ave|Blvd|Street|Road|路|街))', text)
        location = location_match.group(1) if location_match else ""

        # 商品名提取
        products = re.findall(r'[\d\.]+x?[\s\-]*(.*?)[\s]*\$?\d+\.\d{2}', text, re.IGNORECASE)
        if not products:
            products = re.findall(r'[\d\.]+[ ]?(.*?)[:： ]*\$?\d+\.\d{2}', text, re.IGNORECASE)
        items = [p.strip() for p in products if len(p.strip()) >= 2]

        return merchant, date_time, location, total_amount, items
    except Exception as e:
        print("⚠️ 解析失败：", e)
        return "", "", "", None, []

def classify_text(text):
    text_cut = " ".join(jieba.cut(text))
    X_input = vectorizer.transform([text_cut])
    prediction = model.predict(X_input)
    return prediction[0]

def main():
    df = pd.read_csv("F:/ai/ocr_recognition/dataset/train_labels_extracted.csv", encoding='utf-8')
    results = []

    for idx, row in df.iterrows():
        image_path, text = row["image_path"], row["text"]
        merchant, date_time, location, total_amount, items = extract_receipt_info(text)
        category = classify_text(text)

        results.append({
            "image_path": image_path,
            "merchant": merchant,
            "date_time": date_time,
            "location": location,
            "total_amount": total_amount,
            "items": "; ".join(items),
            "category": category,
            "text": text
        })

    output_df = pd.DataFrame(results)
    output_df.to_csv("F:/ai/ocr_recognition/dataset/structured_receipts_with_category.csv", index=False, encoding='utf-8-sig')
    print("✅ 完整分类结构化结果保存成功！")

if __name__ == "__main__":
    main()
