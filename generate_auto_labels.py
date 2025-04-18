import pandas as pd
import re

# 分类关键词字典（细化）
CATEGORY_KEYWORDS = {
    "餐饮": {
        "快餐": ["burger", "fries", "mcdonald", "kfc", "pizza", "noodle", "汉堡", "薯条", "披萨", "炸鸡", "炸串"],
        "中餐": ["炒饭", "麻辣烫", "宫保鸡丁", "面条", "包子", "饺子", "小笼包", "烧烤", "火锅", "宫保鸡丁", "麻辣香锅", "红烧肉", "狮子头", "糖醋排骨", "水煮鱼", "水饺", "炸酱面"],
        "西餐": ["steak", "spaghetti", "pasta", "salad", "soup", "牛排", "意大利面", "沙拉", "汤", "披萨", "烤鸡", "烤牛排", "牛肉汉堡"],
        "甜点": ["ice cream", "cake", "pie", "chocolate", "cookies", "tart", "冰淇淋", "蛋糕", "派", "巧克力", "曲奇", "塔", "泡芙"],
        "饮品": ["coffee", "tea", "juice", "饮料", "咖啡", "果汁", "茶", "奶茶", "鲜榨汁", "葡萄酒", "啤酒"],
    },
    "购物": {
        "水果": ["apple", "banana", "orange", "grape", "mango", "strawberry", "水蜜桃", "苹果"],
        "衣物": ["dress", "shoes", "shirt", "pants", "jacket", "tunic", "cover up", "suits", "microkini", "裙子", "裤子"],
        "电子产品": ["phone", "laptop", "charger", "headphones", "tablet", "电池", "耳机", "电视"],
        "化妆品": ["lipstick", "foundation", "eyeliner", "blush", "eyeshadow", "口红", "粉底"]
    },
    "交通": ["uber", "taxi", "bus", "metro", "交通"],
    "娱乐": ["movie", "cinema", "netflix", "game", "娱乐"],
    "超市": ["dollar tree", "costco", "walmart", "家乐福", "超市"]
}

# 自动分类函数（支持模糊匹配）
def classify_text(text):
    text = str(text).lower()
    for category, subcategories in CATEGORY_KEYWORDS.items():
        if isinstance(subcategories, dict):  # 如果是餐饮或购物类别，进行子类分类
            for subcategory, keywords in subcategories.items():
                for keyword in keywords:
                    if re.search(rf'\b{re.escape(keyword.lower())}\b', text):
                        return f"{category} - {subcategory}"  # 返回更细化的子类
        else:
            for keyword in subcategories:
                if re.search(rf'\b{re.escape(keyword.lower())}\b', text):
                    return category
    return "其他"  # 没有匹配的归类为“其他”

# 读取原始 OCR 文本数据
input_path = r'F:\ai\ocr_recognition\dataset\train_labels.csv'
output_path = r'F:\ai\ocr_recognition\dataset\train_labels_labeled.csv'

df = pd.read_csv(input_path)

# 自动分类
df['category'] = df['text'].apply(classify_text)

# 保存新文件
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"✅ 自动分类完成！已保存至：{output_path}")
