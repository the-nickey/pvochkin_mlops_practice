#!/bin/bash

# Запустим скрипт, который создаёт данные
python data_creation.py

# Запустим скрипт, который производит предобработку
python model_preprocessing.py

# Теперь обучим модель
python model_preparation.py

# Протестируем модель - получим метрики качества
python model_testing.py
