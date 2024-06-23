import numpy as np
import pandas as pd
import random
import os

def create_dataset(size, noise_level=0, anomalies=False):
    """Создает набор данных с заданным уровнем шума и аномалий."""
    
    # Генерируем данные о температуре
    time = np.arange(size)
    temperature = 15 + 2 * np.sin(time / 10)

    # Добавляем шум
    noise = noise_level * np.random.randn(size)
    temperature += noise

    # Добавляем аномалии
    if anomalies:
        anomaly_indices = random.sample(range(size), int(0.05 * size))  # 5% аномалий
        temperature[anomaly_indices] = np.random.randint(0, 30, size=len(anomaly_indices))

    # Создаем DataFrame
    data = pd.DataFrame({'time': time, 'temperature': temperature})
    
    return data

if __name__ == "__main__":
    os.makedirs("train", exist_ok=True)
    os.makedirs("test", exist_ok=True)

    # Создаем несколько наборов данных с разными параметрами
   
    print ('Genereting datasets...')
    for i in range(3):
        # Набор данных с шумом
        data = create_dataset(100, noise_level=0.5)
        data.to_csv(f"train/dataset_{i}_noise.csv", index=False)
        
        # Набор данных с аномалиями
        data = create_dataset(100, anomalies=True)
        data.to_csv(f"train/dataset_{i}_anomaly.csv", index=False)

    # Создаем тестовые наборы данных
    for i in range(3):
        data = create_dataset(50, noise_level=0.2)
        data.to_csv(f"test/dataset_{i}_test.csv", index=False)
