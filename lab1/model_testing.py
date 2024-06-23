import pandas as pd
from sklearn.metrics import mean_squared_error
from model_preparation import prepare_and_train_model

def test_model(model, data_path):
    """Проверяет модель машинного обучения."""
    
    data = pd.read_csv(data_path)
    print ('Model testing...')
    # Делаем прогноз
    y_pred = model.predict(data[['time']])
    
    # Вычисляем метрику ошибки
    mse = mean_squared_error(data['temperature'], y_pred)
    print(f"MSE: {mse}")

if __name__ == "__main__":
    # Пример использования
    data_path = "test/dataset_0_test.csv"
    model = prepare_and_train_model("train/dataset_0_noise.csv")
    test_model(model, data_path)
