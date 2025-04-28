import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path):
    """加载数据集"""
    try:
        data = pd.read_csv(file_path)
        print("\n数据集信息:")
        print(f"行数: {len(data)}")
        print(f"列数: {len(data.columns)}")
        print("\n列名:")
        print(data.columns.tolist())
        print("\n数据预览:")
        print(data.head())
        return data
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None

def preprocess_data(data):
    """数据预处理"""
    # 分离特征和目标变量，排除ID列
    X = data.drop(['DS', 'ID'], axis=1)  # 排除目标变量DS和ID列
    y = data['DS']
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def apply_smote(X, y):
    """应用SMOTE过采样"""
    print("\n原始数据集中各类别的样本数:")
    print(pd.Series(y).value_counts().sort_index())
    
    # 创建SMOTE对象
    smote = SMOTE(random_state=42)
    
    # 应用SMOTE过采样
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print("\nSMOTE过采样后各类别的样本数:")
    print(pd.Series(y_resampled).value_counts().sort_index())
    
    return X_resampled, y_resampled

def train_naive_bayes(X, y):
    """训练朴素贝叶斯模型"""
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 对训练集应用SMOTE
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    
    # 创建并训练模型
    nb_model = GaussianNB()
    nb_model.fit(X_train_resampled, y_train_resampled)
    
    # 预测
    y_pred = nb_model.predict(X_test)
    
    return nb_model, X_train, X_test, y_train, y_test, y_pred

def evaluate_model(y_test, y_pred):
    """评估模型性能"""
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # 获取分类报告
    class_report = classification_report(y_test, y_pred)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    
    return conf_matrix, class_report, accuracy

def plot_confusion_matrix(conf_matrix):
    """绘制混淆矩阵热力图"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix.png')
    plt.show()
    plt.close()

def plot_feature_importance(model, feature_names):
    """绘制特征重要性图"""
    plt.figure(figsize=(12, 6))
    
    # 计算每个特征的平均方差
    feature_vars = np.mean(model.var_, axis=0)
    
    # 创建特征重要性数据框
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Variance': feature_vars
    })
    
    # 按方差排序
    feature_importance = feature_importance.sort_values('Variance', ascending=True)
    
    # 创建水平条形图
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance, y='Feature', x='Variance')
    plt.title('Feature Importance Analysis (Based on Variance)')
    plt.xlabel('Average Variance')
    plt.ylabel('Feature Name')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()
    plt.close()

def main():
    # 加载数据
    data = load_data('earthquake_damage.csv')
    if data is None:
        return
    
    # 数据预处理
    X_scaled, y = preprocess_data(data)
    
    # 划分训练集和测试集（20%测试集，80%训练集）
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # 确保各类别比例一致
    )
    
    print("\n数据集分割信息:")
    print(f"原始数据集大小: {len(y)}")
    print(f"训练集大小: {len(y_train)}")
    print(f"测试集大小: {len(y_test)}")
    print("\n训练集中各类别的样本数:")
    print(pd.Series(y_train).value_counts().sort_index())
    
    # 对训练集应用SMOTE
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    
    # 创建并训练模型
    nb_model = GaussianNB()
    nb_model.fit(X_train_resampled, y_train_resampled)
    
    # 预测
    y_pred = nb_model.predict(X_test)
    
    # 评估模型
    conf_matrix, class_report, accuracy = evaluate_model(y_test, y_pred)
    
    # 打印结果
    print("\n模型评估结果:")
    print(f"\n准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(class_report)
    
    # 绘制可视化图表
    plot_confusion_matrix(conf_matrix)
    # 获取特征名称（排除DS和ID列）
    feature_names = data.drop(['DS', 'ID'], axis=1).columns
    plot_feature_importance(nb_model, feature_names)

if __name__ == "__main__":
    main() 