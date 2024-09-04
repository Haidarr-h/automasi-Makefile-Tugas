import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

def train_model():
    train_data = pd.read_csv('data/train_data.csv')
    X_train = train_data.drop(columns=['target'])
    y_train = train_data['target']

    model = LogisticRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, 'models/model.pkl')

if __name__ == "__main__":
    train_model()
