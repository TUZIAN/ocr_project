import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

# 加载模型和向量化器
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

# 训练和评估模型
def auto_train_and_evaluate(data, labels, num_epochs=5, save_interval=2):
    best_accuracy = 0
    model, vectorizer = load_model()

    # 如果模型不存在，训练一个新的模型
    if model is None or vectorizer is None:
        print("没有找到现有模型，开始训练新模型")
        from train_text_classifier import train_classifier  # 修改为正确的文件名
        model, vectorizer = train_classifier(data, labels)

    for epoch in range(num_epochs):
        print(f"训练轮次 {epoch + 1}/{num_epochs}")

        # 使用训练数据进行训练
        X = vectorizer.transform(data)  # 将文本数据转化为向量
        X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.2, random_state=42)

        model.fit(X_train, y_train)

        # 评估模型
        y_pred = model.predict(X_val)
        accuracy = classification_report(y_val, y_pred, output_dict=True, zero_division=1)['accuracy']
        print(f"当前验证集准确率：{accuracy:.4f}")

        # 如果当前准确率比之前的好，保存模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            with open('best_model.pkl', 'wb') as f:
                pickle.dump(model, f)
            with open('vectorizer.pkl', 'wb') as f:
                pickle.dump(vectorizer, f)
            print(f"模型已保存，当前最佳准确率：{best_accuracy:.4f}")

# 主函数
def main():
    # 载入数据集
    df = pd.read_csv("F:/ai/ocr_recognition/dataset/train_labels_extracted.csv")
    data = df['text']  # 小票文本
    labels = df['category']  # 分类标签

    # 输入训练轮次
    num_epochs = int(input("请输入训练轮次："))

    # 自动训练和评估
    auto_train_and_evaluate(data, labels, num_epochs=num_epochs)

if __name__ == "__main__":
    main()
