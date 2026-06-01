# ML Preprocessing Research

## Описание проекта

Проект исследует влияние методов предобработки данных на качество моделей машинного обучения на двух датасетах:

- Adult Census Income;
- Breast Cancer Wisconsin Diagnostic Dataset.

Цель работы - показать, как импутация, масштабирование, кодирование категориальных признаков и балансировка классов влияют на Accuracy, F1-score и ROC-AUC в разных типах задач бинарной классификации.

## Что исследуется

- обработка пропусков;
- масштабирование числовых признаков;
- кодирование категориальных признаков;
- балансировка классов;
- сравнение моделей на разных структурах данных.

## Используемые методы

- SimpleImputer;
- KNNImputer;
- StandardScaler;
- MinMaxScaler;
- OneHotEncoder;
- class_weight;
- SMOTE.

## Модели

- Logistic Regression;
- SVC;
- Random Forest;
- MLPClassifier.

## Метрики

- Accuracy;
- F1-score;
- ROC-AUC.

Для Adult Census положительный класс для F1-score: `>50K`.
Для Breast Cancer положительный класс для F1-score: `malignant`.

## Используемые технологии

- Python 3.13
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn
- Jupyter Notebook
- Git
- GitHub

## Полученные навыки

- Проведение Exploratory Data Analysis (EDA)
- Подготовка и очистка данных
- Обработка пропущенных значений
- Масштабирование признаков
- Балансировка классов
- Построение Pipeline и ColumnTransformer
- Оценка качества моделей машинного обучения
- Сравнение моделей на разных датасетах
- Проведение кросс-датасетного анализа
- Работа с Git и GitHub

## Структура проекта

- `data/` - исходные данные проекта;
- `notebooks/` - EDA, baseline и экспериментальные ноутбуки;
- `src/` - общие функции загрузки данных, метрик и сохранения артефактов;
- `results/` - CSV-таблицы с результатами экспериментов;
- `figures/` - графики, сохраненные из ноутбуков;
- `report/` - материалы для итогового отчета.

## Ноутбуки

1. `notebooks/01_eda.ipynb` - EDA Adult Census.
2. `notebooks/02_baseline.ipynb` - baseline Adult Census.
3. `notebooks/03_scaling_experiment.ipynb` - scaling experiment Adult Census.
4. `notebooks/04_imputation_experiment.ipynb` - imputation experiment Adult Census.
5. `notebooks/05_balancing_experiment.ipynb` - balancing experiment Adult Census.
6. `notebooks/06_models_comparison.ipynb` - models comparison Adult Census.
7. `notebooks/07_final_analysis.ipynb` - final analysis Adult Census.
8. `notebooks/08_breast_cancer_eda.ipynb` - EDA Breast Cancer.
9. `notebooks/09_breast_cancer_baseline.ipynb` - baseline Breast Cancer.
10. `notebooks/10_breast_cancer_scaling_experiment.ipynb` - scaling experiment Breast Cancer.
11. `notebooks/11_breast_cancer_imputation_experiment.ipynb` - imputation experiment Breast Cancer.
12. `notebooks/12_breast_cancer_models_comparison.ipynb` - models comparison Breast Cancer.
13. `notebooks/13_cross_dataset_analysis.ipynb` - cross-dataset analysis.

## Основные результаты

### Adult Census

Baseline Logistic Regression:

| Accuracy | F1-score | ROC-AUC |
|---:|---:|---:|
| 0.8520 | 0.6539 | 0.9012 |

Лучшие результаты Adult Census:

| Критерий | Метод / модель | Значение |
|---|---|---:|
| Лучший F1-score | Logistic Regression + SMOTE | 0.6762 |
| Лучший ROC-AUC | Logistic Regression + class_weight="balanced" | 0.9042 |
| Лучшая Accuracy | Logistic Regression + StandardScaler | 0.8525 |

### Breast Cancer

Baseline Logistic Regression:

| Accuracy | F1-score | ROC-AUC |
|---:|---:|---:|
| 0.9386 | 0.9114 | 0.9937 |

Лучшие результаты Breast Cancer:

| Критерий | Метод / модель | Значение |
|---|---|---:|
| Лучший F1-score | Logistic Regression + StandardScaler | 0.9639 |
| Лучший ROC-AUC | Logistic Regression + MinMaxScaler | 0.9997 |
| Лучшая Accuracy | Logistic Regression + StandardScaler | 0.9737 |

## Итоговые выводы

На Adult Census основными сложностями были категориальные признаки, пропуски и дисбаланс классов. Поэтому качество заметно зависит от корректной обработки категорий, импутации и балансировки классов.

На Breast Cancer все признаки числовые, пропуски отсутствуют, а модели показывают более высокое качество. Для этого датасета масштабирование имеет большее значение, потому что признаки измеряются в разных диапазонах.

Импутация является обязательным безопасным шагом preprocessing pipeline, но более сложный KNNImputer не всегда улучшает качество. Главный вывод проекта: качество моделей зависит не только от выбранного алгоритма, но и от корректно построенной цепочки предобработки данных.

## Как запустить проект

1. Создать virtual environment.
2. Установить зависимости:

```bash
pip install -r requirements.txt
```

3. Запускать ноутбуки по порядку из папки `notebooks/`.

Все методы предобработки реализованы внутри `Pipeline`, `ColumnTransformer` или `imblearn.pipeline.Pipeline`, чтобы избежать data leakage.
