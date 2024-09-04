import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

def evaluate_model():
    test_data = pd.read_csv('data/test_data.csv')
    model = joblib.load('models/model.pkl')

    X_test = test_data.drop(columns=['target'])
    y_test = test_data['target']

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    with open('results/evaluation.txt', 'w') as f:
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Precision: {precision}\n')
        f.write(f'Recall: {recall}\n')
        f.write(f'F1 Score: {f1}\n')

if __name__ == "__main__":
    evaluate_model()
