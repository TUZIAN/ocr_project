from paddleocr import PaddleOCR
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# 图片文件夹路径
image_dir = Path(r"F:\ai\ocr_recognition\dataset\train\images")
# 保存的 CSV 路径
output_csv = Path(r"F:\ai\ocr_recognition\dataset\train_labels.csv")

# 初始化 OCR（中英文）
ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # 支持中英文

# 获取所有图片路径（.jpg 和 .png）
image_paths = list(image_dir.rglob("*.jpg")) + list(image_dir.rglob("*.png"))
print(f"共发现 {len(image_paths)} 张小票图片，正在识别中...")

# 自动识别并生成文本
data = []
for img_path in tqdm(image_paths):
    result = ocr.ocr(str(img_path))
    full_text = ""
    for line in result:
        for word_info in line:
            full_text += word_info[1][0] + " "
    data.append({
        "image_path": str(img_path),
        "text": full_text.strip()
    })

# 保存 CSV 文件
df = pd.DataFrame(data)
df.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"\n✅ 标注完成！标注结果保存在：{output_csv}")
