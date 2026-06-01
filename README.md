# ML Preprocessing Research

## Описание проекта

Исследование влияния методов предобработки данных на качество моделей машинного обучения на примере датасета Adult Census.

## Цель исследования

Оценить, как методы импутации, масштабирования и балансировки классов влияют на Accuracy, F1-score и ROC-AUC моделей машинного обучения.

## Используемый датасет

Adult Census Income Dataset. Целевая переменная: `class`, положительный класс для F1-score: `>50K`.

## Методы предобработки

- SimpleImputer
- KNNImputer
- StandardScaler
- MinMaxScaler
- OneHotEncoder
- class_weight
- SMOTE

## Модели

- Logistic Regression
- SVC
- Random Forest
- MLPClassifier

## Метрики

- Accuracy
- F1-score
- ROC-AUC

## Структура проекта

- `data/` - исходные данные проекта.
- `notebooks/` - EDA, baseline и экспериментальные ноутбуки.
- `src/` - общие функции загрузки данных, метрик и сохранения артефактов.
- `results/` - CSV-таблицы с результатами экспериментов.
- `figures/` - графики, сохраненные из ноутбуков.
- `report/` - материалы для итогового отчета.

## Основные результаты

Baseline Logistic Regression:

| Accuracy | F1-score | ROC-AUC |
|---:|---:|---:|
| 0.8520 | 0.6539 | 0.9012 |

Лучшие результаты в экспериментах Adult Census:

| Критерий | Метод / модель | Значение |
|---|---|---:|
| Лучший F1-score | Logistic Regression + SMOTE | 0.6762 |
| Лучший ROC-AUC | Logistic Regression + class_weight="balanced" | 0.9042 |
| Лучшая Accuracy | Logistic Regression + StandardScaler | 0.8525 |

Вывод: preprocessing влияет на качество моделей, но эффект зависит от метрики. При дисбалансе классов Accuracy не должна быть единственной метрикой; балансировка может снижать Accuracy, но улучшать F1-score для класса `>50K`.

## Как запустить проект

1. Создать virtual environment.
2. Установить зависимости:

```bash
pip install -r requirements.txt
```

3. Запустить notebooks по порядку:

```text
notebooks/01_eda.ipynb
notebooks/02_baseline.ipynb
notebooks/03_scaling_experiment.ipynb
notebooks/04_imputation_experiment.ipynb
notebooks/05_balancing_experiment.ipynb
notebooks/06_models_comparison.ipynb
notebooks/07_final_analysis.ipynb
```

Все методы предобработки реализованы внутри `Pipeline` и `ColumnTransformer`. SMOTE используется только внутри `imblearn.pipeline.Pipeline` после preprocessing и до classifier.
