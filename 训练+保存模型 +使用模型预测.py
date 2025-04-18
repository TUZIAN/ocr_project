import os
import re
import jieba
import joblib
import pandas as pd
from tkinter import Tk, filedialog
from paddleocr import PaddleOCR
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 📍 模型与数据路径
MODEL_PATH = "F:/ai/ocr_recognition/models/receipt_classifier_svm.pkl"
DATA_PATH = "F:/ai/ocr_recognition/dataset/train_labels_extracted.csv"

# 🚩 关键词映射分类标签
category_keywords = {
    "餐饮": ["米饭", "炒饭", "面", "饮料", "饭", "鸡", "牛肉", "套餐", "快餐", "noodle", "rice", "meal", "burger", "soup", "drink", "pizza", "pasta", "coffee", "starbucks", "mcdonald's", "kfc", "sandwich", "taco", "salad", "sushi"],
    "购物": ["衣服", "鞋", "裤", "裙", "t-shirt", "clothes", "shoes", "dress", "bag", "jacket", "watch", "jeans", "sneakers", "outfit", "sweater", "nike", "adidas", "h&m", "zara", "uniqlo", "gucci", "lv", "prada", "chanel"],
    "交通": ["地铁", "公交", "出租车", "高铁", "metro", "bus", "taxi", "transportation", "subway", "train", "airline", "flight", "uber", "lyft", "bolt", "grab", "car rental", "bus ticket"],
    "医疗": ["药", "医院", "体检", "诊费", "medicine", "clinic", "hospital", "pharmacy", "doctor", "prescription", "dentist", "vaccination", "medical", "healthcare", "surgery"],
    "教育": ["培训", "学费", "课程", "education", "tuition", "class", "school", "university", "online course", "workshop", "degree", "certification", "exam", "study materials", "language course", "esl"],
    "娱乐": ["电影", "游戏", "KTV", "影票", "cinema", "game", "movie", "ticket", "concert", "show", "theater", "theme park", "sports event", "netflix", "playstation", "xbox", "nintendo", "event ticket", "party", "club"],
    "外卖": ["delivery", "rider", "Deliveroo", "Meituan", "Ele.me", "外卖", "配送单", "外卖单", "uber eats", "grubhub", "door dash", "postmates", "food delivery", "takeout", "just eat", "seamless", "food panda", "zomato"]
}

# 🧠 OCR 实例（中英文）
ocr_ch = PaddleOCR(use_angle_cls=True, lang='ch')
ocr_en = PaddleOCR(use_angle_cls=True, lang='en')

# ✂️ 分词
def tokenize(text):
    return " ".join(jieba.cut(str(text)))

# 🧾 OCR 识别函数
def ocr_recognize_receipt(image_path):
    result_ch = ocr_ch.ocr(image_path, cls=True)
    result_en = ocr_en.ocr(image_path, cls=True)

    text_ch = " ".join([line[1][0] for line in result_ch[0]])
    print(f"中文OCR结果: {text_ch}")

    text_en = " ".join([line[1][0] for line in result_en[0]])
    print(f"英文OCR结果: {text_en}")

    combined_text = text_ch + " " + text_en
    return combined_text.strip()

# 🏷️ 分类函数（关键词）
def classify_receipt(text):
    all_text = text.lower()
    category = "其他"
    for cat, keywords in category_keywords.items():
        if any(keyword.lower() in all_text for keyword in keywords):
            category = cat
            break
    return category

# 💡 优化字段提取（金额、商户）
def extract_info_from_text(text):
    lines = text.strip().split('\n') if '\n' in text else text.strip().split()
    merchant = "未知"
    amount = "未知"

    # 商户提取
    merchant_keywords = ['店', '公司', '超市', '便利店', '餐厅', '商场', 'food', 'shop', 'store', 'restaurant', 'mart']
    for line in lines[:5]:  # 前5行中查找商户
        if any(kw.lower() in line.lower() for kw in merchant_keywords):
            merchant = line.strip()
            break
    if merchant == "未知" and lines:
        merchant = lines[0]

    # 金额提取（优先识别关键词行）
    amount_keywords = ['总计', '合计', '应收', '实收', '消费金额', '金额', 'total', 'amount', 'paid', 'payable', 'price']
    for line in reversed(lines):  # 从后往前找
        if any(kw.lower() in line.lower() for kw in amount_keywords):
            found = re.findall(r'[\d]+(?:\.\d{1,2})?', line)
            if found:
                amount = found[-1]
                break

    # 兜底提取最大金额
    if amount == "未知":
        all_amounts = re.findall(r'[\d]+(?:\.\d{1,2})?', text)
        if all_amounts:
            amount = max(all_amounts, key=lambda x: float(x))

    return amount, merchant

# 🔧 模型训练与保存
def train_and_save_model():
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
    df = df[df["category"] != "其他"]

    texts = [tokenize(t) for t in df["text"].astype(str).tolist()]
    labels = df["category"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    clf_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=2000)),
        ('clf', SVC(kernel='linear'))
    ])

    clf_pipeline.fit(X_train, y_train)
    y_pred = clf_pipeline.predict(X_test)
    print("📊 分类报告：\n", classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf_pipeline, MODEL_PATH)
    print(f"✅ 模型已保存至：{MODEL_PATH}")

# 🧠 加载模型并预测分类
def classify_new_receipt(text):
    if not os.path.exists(MODEL_PATH):
        print("❌ 模型文件不存在，请先训练模型")
        return "未分类"
    model = joblib.load(MODEL_PATH)
    text_cut = tokenize(text)
    return model.predict([text_cut])[0]

# 📤 上传图片并进行分类
def upload_and_classify():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="选择小票图片", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    if file_path:
        ocr_text = ocr_recognize_receipt(file_path)
        category = classify_new_receipt(ocr_text)
        amount, merchant = extract_info_from_text(ocr_text)
        print(f"\n🧾 分类：{category} ｜ 💰 金额：{amount} ｜ 🏬 商户：{merchant}")

# 🚪 程序入口
if __name__ == "__main__":
    train_and_save_model()
    upload_and_classify()
