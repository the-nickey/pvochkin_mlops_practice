import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def prepare_and_train_model(data_path):
    """Создает и обучает модель машинного обучения."""
    
    data = pd.read_csv(data_path)
    
    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(data[['time']], data['temperature'], test_size=0.2)
    
    # Создаем модель линейной регрессии
    model = LinearRegression()
    
    # Обучаем модель
    model.fit(X_train, y_train)

    return model

if __name__ == "__main__":
    # Пример использования
    data_path = "train/dataset_0_noise.csv"
    print ('Model preparation...')
    model = prepare_and_train_model(data_path)
    print(f"intercept: {model.intercept_}")
    print(f"slope: {model.coef_}")
