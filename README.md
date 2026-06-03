# ML Preprocessing Research

Исследовательский проект о влиянии методов предобработки данных на качество моделей бинарной классификации.

В проекте сравниваются два датасета:

- **Adult Census Income**: табличные социально-демографические данные с числовыми и категориальными признаками, пропусками и дисбалансом классов.
- **Breast Cancer Wisconsin Diagnostic**: медицинский датасет с числовыми признаками опухолей и бинарной целевой переменной.

Главная идея проекта: качество модели зависит не только от выбранного алгоритма, но и от корректной цепочки preprocessing. Эксперименты вынесены в Jupyter Notebook, а итоговые таблицы и графики сохранены в `results/` и `figures/`.

## Что исследуется

- обработка пропущенных значений;
- масштабирование числовых признаков;
- кодирование категориальных признаков;
- балансировка классов на Adult Census;
- сравнение нескольких моделей на одинаковых схемах предобработки;
- сопоставление выводов между двумя разными табличными датасетами.

## Датасеты

### Adult Census Income

Файл: `data/raw/adult-census.csv`

Целевая переменная: `class`

Положительный класс для F1-score: `>50K`

Особенности:

- есть числовые и категориальные признаки;
- пропуски представлены как `?` и нормализуются при загрузке;
- присутствует дисбаланс классов;
- категориальные признаки кодируются через `OneHotEncoder`.

### Breast Cancer Wisconsin Diagnostic

Файл: `data/raw/breast-cancer.csv`

В утилитах проекта также используется загрузка через `sklearn.datasets.load_breast_cancer()`.

Целевая переменная: `target`

Положительный класс для F1-score: `malignant`

Особенности:

- все признаки числовые;
- пропуски отсутствуют;
- масштабирование сильнее влияет на линейные и kernel-модели, чем кодирование или импутация.

## Методы и модели

Методы предобработки:

- `SimpleImputer`;
- `KNNImputer`;
- `StandardScaler`;
- `MinMaxScaler`;
- `OneHotEncoder`;
- `class_weight="balanced"`;
- `SMOTE`.

Модели:

- `LogisticRegression`;
- `SVC`;
- `RandomForestClassifier`;
- `MLPClassifier`.

Метрики:

- `Accuracy`;
- `F1-score`;
- `ROC-AUC`.

## Структура проекта

```text
.
├── data/
│   └── raw/                         # исходные CSV-датасеты
├── figures/                         # PNG-графики из ноутбуков
├── notebooks/                       # EDA и экспериментальные ноутбуки
├── results/                         # CSV-таблицы с метриками
├── src/
│   └── experiment_utils.py          # общие функции загрузки, метрик и сохранения
├── requirements.txt                 # зависимости окружения
└── README.md
```

## Ноутбуки

Рекомендуемый порядок запуска:

1. `notebooks/01_adult_eda.ipynb` - EDA для Adult Census.
2. `notebooks/02_adult_baseline.ipynb` - baseline-модель для Adult Census.
3. `notebooks/03_adult_scaling_experiment.ipynb` - сравнение масштабирования для Adult Census.
4. `notebooks/04_adult_imputation_experiment.ipynb` - сравнение импутации для Adult Census.
5. `notebooks/05_adult_balancing_experiment.ipynb` - сравнение балансировки классов для Adult Census.
6. `notebooks/06_adult_models_comparison.ipynb` - сравнение моделей для Adult Census.
7. `notebooks/07_adult_final_analysis.ipynb` - итоговый анализ Adult Census.
8. `notebooks/08_breast_cancer_eda.ipynb` - EDA для Breast Cancer.
9. `notebooks/09_breast_cancer_baseline.ipynb` - baseline-модель для Breast Cancer.
10. `notebooks/10_breast_cancer_scaling_experiment.ipynb` - сравнение масштабирования для Breast Cancer.
11. `notebooks/11_breast_cancer_imputation_experiment.ipynb` - сравнение импутации для Breast Cancer.
12. `notebooks/12_breast_cancer_models_comparison.ipynb` - сравнение моделей для Breast Cancer.
13. `notebooks/13_cross_dataset_analysis.ipynb` - кросс-датасетный анализ.

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
| Лучший ROC-AUC | Logistic Regression + `class_weight="balanced"` | 0.9042 |
| Лучшая Accuracy | Logistic Regression + StandardScaler / MinMaxScaler | 0.8525 |

В сравнении моделей на Adult Census лучший F1-score показал `RandomForestClassifier` со значением `0.6631`, а лучший ROC-AUC остался у `LogisticRegression` со значением `0.9040`.

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

В сравнении моделей на Breast Cancer лучший набор метрик также у `LogisticRegression` с `SimpleImputer + StandardScaler`: Accuracy `0.9737`, F1-score `0.9639`, ROC-AUC `0.9960`.

## Итоговые выводы

На Adult Census основными источниками сложности являются категориальные признаки, пропуски и дисбаланс классов. Балансировка снижает Accuracy, но улучшает F1-score для положительного класса `>50K`, что важно при фокусе на менее представленном классе.

На Breast Cancer признаки уже числовые и без пропусков, поэтому основное влияние дает масштабирование. `StandardScaler` улучшает Accuracy и F1-score, а `MinMaxScaler` дает максимальный ROC-AUC.

Импутация остается обязательным безопасным шагом pipeline, но более сложный `KNNImputer` в этих экспериментах не дал прироста относительно простых стратегий. Все основные схемы предобработки реализованы через `Pipeline`, `ColumnTransformer` или `imblearn.pipeline.Pipeline`, чтобы снизить риск data leakage.

## Как запустить проект

1. Создать и активировать виртуальное окружение:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
source .venv/bin/activate
```

2. Установить зависимости:

```bash
pip install -r requirements.txt
```

3. Запустить Jupyter:

```bash
jupyter notebook
```

4. Открывать ноутбуки из папки `notebooks/` в указанном выше порядке.

## Ключевые артефакты

- `results/cross_dataset_summary.csv` - объединенная таблица результатов по двум датасетам.
- `results/final_summary_results.csv` - итоговая таблица экспериментов Adult Census.
- `results/breast_cancer_*_results.csv` - результаты экспериментов Breast Cancer.
- `figures/final_summary_metrics.png` - итоговая визуализация Adult Census.
- `figures/cross_dataset_*.png` - графики кросс-датасетного сравнения.
- `src/experiment_utils.py` - общие функции для загрузки данных, расчета метрик и сохранения артефактов.
