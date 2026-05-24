# Credit Default Prediction
**Kaggle: [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)**


## Task

Бинарная классификация: предсказать, допустит ли заёмщик просрочку **90+ дней** в течение двух лет (`SeriousDlqin2yrs`).

Основная метрика — **ROC-AUC**. Дополнительно отслеживаются PR-AUC, Precision, Recall и F1, поскольку классы сильно несбалансированы (93.3% / 6.7%).


## Data

| | |
|---|---|
| Источник | Kaggle: Give Me Some Credit |
| Наблюдений | 150 000 |
| Признаков | 10 |
| Дефолтов | 6.7% |


## Features

| Feature | Type | Description |
|---|---|---|
| `age` | int | Возраст заёмщика |
| `NumberOfDependents` | int | Количество иждивенцев (дети, супруг и др.) |
| `MonthlyIncome` | float | Ежемесячный доход |
| `DebtRatio` | float | Долговые обязательства / доход. При нулевом/пропущенном доходе содержит абсолютную сумму долга — признак частично некорректен |
| `RevolvingUtilizationOfUnsecuredLines` | float | Баланс по беззалоговым кредитным линиям / кредитный лимит |
| `NumberOfOpenCreditLinesAndLoans` | int | Общее количество открытых кредитов и кредитных линий |
| `NumberRealEstateLoansOrLines` | int | Количество кредитов под залог недвижимости (ипотека и др.) |
| `NumberOfTime30-59DaysPastDueNotWorse` | int | Количество просрочек 30–59 дней |
| `NumberOfTime60-89DaysPastDueNotWorse` | int | Количество просрочек 60–89 дней |
| `NumberOfTimes90DaysLate` | int | Количество просрочек 90+ дней |



## Results

### Метрики на тестовой выборке (threshold = 0.5)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| catboost | 0.2179 | 0.7746 | 0.3401 | **0.8647** | **0.3994** |
| random_forest | 0.2510 | 0.7202 | **0.3723** | 0.8628 | 0.3927 |
| logreg | 0.2243 | 0.7431 | 0.3446 | 0.8585 | 0.3764 |
| knn | 0.5930 | 0.1686 | 0.2625 | 0.8491 | 0.3749 |
| baseline_logreg | 0.1677 | 0.6404 | 0.2658 | 0.7831 | 0.3028 |

**Выводы:**
- CatBoost, RF и LogReg находятся в диапазоне ROC-AUC 0.858–0.865 — разрыв несущественен; бустинг не даёт принципиального выигрыша перед линейной моделью.
- kNN структурно не подходит для задачи: при threshold=0.5 выдаёт Recall=0.17, пропуская 83% дефолтов.
- RF даёт лучший F1 (0.37); CatBoost — лучший Recall (0.77) и PR-AUC (0.40).
- ⚠️ Метрики при threshold=0.5 ориентировочны. Ошибки FN и FP имеют разную стоимость — требуется тюнинг порога через cost-функцию с бизнес-весами.

## Key Feature Insights (SHAP)

- **`RevolvingUtilizationOfUnsecuredLines`** — главный предиктор дефолта во всех моделях.
- **`age`** стабильно входит в топ-5 у tree-моделей: моложе → выше риск (подтверждается EDA).
- **`MonthlyIncome`** и **`DebtRatio`** имеют низкую важность — само значение дохода слабо разделяет классы.



## Methodology

### EDA
- Выдвинуты и проверены гипотезы о зависимостях признаков друг с другом и с таргетом.
- Выявлены аномальные значения, пропуски, искажения в данных
- Определены признаки с нелинейной связью с таргетом (`age`, `NumberOfOpenCreditLinesAndLoans`).

**Основные результаты:**  
- `MonthlyIncome`: 19.8% пропусков; нулевые значения содержательно эквивалентны NaN.
- `NumberOfDependents`: 2.6% пропусков; медиана = 0, заполняется медианой.
- `DebtRatio`: при нулевом/пропущенном доходе поле содержит абсолютную сумму долга вместо ratio.
- `RevolvingUtilizationOfUnsecuredLines`: значения > 2 — вероятные ошибки записи; риск дефолта резко возрастает при превышении лимита (> 1.0).
- `NumberOfTime*`: значения 96 и 98 — технические маркеры, не реальные числа просрочек.

### Feature Engineering
Добавлены 17 производных признаки поверх исходных:

| Признак | Описание |
|---|---|
| `TotalLatePayments` | Суммарное количество просрочек всех типов |
| `WeightedDelinquency` | Взвешенная серьёзность просрочек (1×N30 + 2×N60 + 3×N90) |
| `MaxDelinquencySeverity` | Максимальный уровень серьёзности просрочки (0–3) |
| `monthly_debt_payment` | `DebtRatio × MonthlyIncome` (абсолютный долг) |
| `income_per_dependent` | Доход на иждивенца |
| `total_credit_lines` | Сумма всех кредитных линий |
| `revutil_over_limit` | Флаг: `RevolvingUtilization > 1.0` |
| `has_real_estate_loan` | Флаг наличия ипотеки / залогового кредита |
| `income_missing` | Флаг отсутствия данных о доходе |
| `is_young` | Флаг возраста < 25 лет |
| `no_open_lines` | Флаг нулевого количества открытых кредитных линий |  
...

### Preprocessing Pipeline

Каждая модель получает собственный препроцессинг через фабрику `build_preprocessor(fe, clip, log, scale)`.

| Шаг | Описание |
|---|---|
| `FeatureEngineeringTransformer` (fe) | Добавляет инженерные признаки |
| `SimpleImputer` | Медианная импутация |
| `ColumnClipper` | Клиппинг по p99 (только числовые, не бинарные) |
| `Log1pTransformer` | `log1p` трансформация скошенных признаков |
| `SelectiveScaler` | `StandardScaler` (только числовые, не бинарные) |


### Modeling

| Модель | Конфигурация |
|---|---|
| `baseline_logreg` | Без FE, дефолтные параметры, StandardScaler, SimpleImputer |
| `logreg` | fe, imputer, clip, log, scale |
| `knn` | fe, imputer, clip, log, scale |
| `random_forest` | fe, imputer, clip |
| `catboost` | fe, imputer |

