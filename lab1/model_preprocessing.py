import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data_path):
    """Выполняет предобработку данных."""
    
    data = pd.read_csv(data_path)
    
    # Масштабирование данных
    scaler = StandardScaler()
    data['temperature'] = scaler.fit_transform(data[['temperature']])
    
    return data

if __name__ == "__main__":
    # Пример использования
    print ('Scaling datasets...')
    data_path = "train/dataset_0_noise.csv"
    preprocessed_data = preprocess_data(data_path)
    print(preprocessed_data.head())
