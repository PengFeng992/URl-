from sklearn import metrics

def calculate_metrics(y_true, y_pred, y_score=None):
    # 计算准确率
    accuracy = metrics.accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)

    # 计算召回率
    recall = metrics.recall_score(y_true, y_pred)
    print("Recall:", recall)

    # 计算精确率
    precision = metrics.precision_score(y_true, y_pred)
    print("Precision:", precision)

    # 计算F1值
    f1 = metrics.f1_score(y_true, y_pred)
    print("F1 Score:", f1)

    # 计算混淆矩阵
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", confusion_matrix)

    # 计算分类报告
    classification_report = metrics.classification_report(y_true, y_pred)
    print("Classification Report:\n", classification_report)

    # 计算ROC曲线和AUC值
    if y_score is not None:
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        auc = metrics.auc(fpr, tpr)
        print("AUC:", auc)


