import re
import csv
import os
import langdetect
import pandas as pd
from datetime import datetime

# 读取之前的OCR识别结果CSV
input_csv_path = r"F:\ai\ocr_recognition\dataset\train_labels.csv"
output_csv_path = r"F:\ai\ocr_recognition\dataset\train_labels_extracted.csv"

# 分类关键词（中英文混合）
category_keywords = {
    "餐饮": ["米饭", "炒饭", "面", "饮料", "饭", "鸡", "牛肉", "套餐", "快餐", "noodle", "rice", "meal", "burger", "soup", "drink"],
    "购物": ["衣服", "鞋", "裤", "裙", "t-shirt", "clothes", "shoes", "dress", "bag"],
    "交通": ["地铁", "公交", "出租车", "高铁", "metro", "bus", "taxi", "transportation"],
    "医疗": ["药", "医院", "体检", "诊费", "medicine", "clinic", "hospital"],
    "教育": ["培训", "学费", "课程", "education", "tuition", "class"],
    "娱乐": ["电影", "游戏", "KTV", "影票", "cinema", "game", "movie", "ticket"]
}


def detect_language(text):
    try:
        lang = langdetect.detect(text)
        return 'zh' if lang.startswith('zh') else 'en'
    except:
        return 'en'


def extract_fields(text):
    lines = text.strip().split('\n')
    language = detect_language(text)
    merchant, date_time, location, total_amount = "", "", "", ""
    items = []

    # 提取商户（默认第一行）
    if lines:
        merchant = lines[0].strip()

    for line in lines:
        line = line.strip()

        # 日期时间匹配（中英文）
        if not date_time:
            date_match = re.search(r'\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}[日]?[ T]?\d{0,2}:?\d{0,2}:?\d{0,2}', line)
            if date_match:
                date_time = date_match.group(0).replace('年', '-').replace('月', '-').replace('日', '')

        # 地址匹配
        if not location and re.search(r'(地址|门店|location|Address)', line, re.I):
            location = line.split(':')[-1].strip()

        # 金额提取（含total/合计）
        if re.search(r'(total|amount|合计|总计)', line, re.I):
            amt_match = re.search(r'(\d+\.\d{2})', line)
            if amt_match:
                total_amount = amt_match.group(1)

        # 商品项提取（中英文）
        if language == 'en':
            item_match = re.match(r'([A-Za-z\s]+)\s+(\d+)\s+(\d+\.\d{2})', line)
            if item_match:
                name = item_match.group(1).strip()
                quantity = int(item_match.group(2))
                price = float(item_match.group(3))
                items.append({"name": name, "quantity": quantity, "price": price})
        else:
            # 中文格式 如: 香菇肉饭 1 15.0
            item_match = re.match(r'([\u4e00-\u9fa5A-Za-z]+)\s+(\d+)\s+(\d+\.\d{2})', line)
            if item_match:
                name = item_match.group(1).strip()
                quantity = int(item_match.group(2))
                price = float(item_match.group(3))
                items.append({"name": name, "quantity": quantity, "price": price})

    # 自动分类
    all_text = text.lower()
    category = "其他"
    for cat, keywords in category_keywords.items():
        if any(keyword.lower() in all_text for keyword in keywords):
            category = cat
            break

    return merchant, date_time, location, items, total_amount, category


# 读取原始CSV并处理
results = []
df = pd.read_csv(input_csv_path)

for _, row in df.iterrows():
    image_path = row['image_path']
    text = row['text']
    merchant, date_time, location, items, total_amount, category = extract_fields(text)

    results.append({
        "image_path": image_path,
        "text": text,
        "merchant": merchant,
        "date_time": date_time,
        "location": location,
        "items": str(items),
        "total_amount": total_amount,
        "category": category
    })

# 写入新CSV
df_out = pd.DataFrame(results)
df_out.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
print(f"✅ 提取完成，结果保存在：{output_csv_path}")
