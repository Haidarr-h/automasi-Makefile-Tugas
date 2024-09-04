import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data():
    from sklearn.datasets import load_iris
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['target'] = iris.target

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    train_data[iris.feature_names] = scaler.fit_transform(train_data[iris.feature_names])
    test_data[iris.feature_names] = scaler.transform(test_data[iris.feature_names])

    train_data.to_csv('data/train_data.csv', index=False)
    test_data.to_csv('data/test_data.csv', index=False)

if __name__ == "__main__":
    load_and_prepare_data()
