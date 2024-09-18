import pandas as pd

from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sklearn.metrics as sk_metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc



# Чтение данных из CSV файла
df = pd.read_csv('Marvel Vs DC.csv')

# Удаление строк с отсутствующими значениями
df = df.dropna()

# Удаление символов "-" из столбца с годом
df['Year'] = df['Year'].str.replace(r'-', '')

# Определение признаков и меток (фильмы и их категория Marvel или DC)
X = df['Movie']
y = df['Category'].map({'Marvel': 0, 'DC': 1})

# Преобразование текста фильмов в числовые признаки с помощью TF-IDF векторов
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

# Разделение данных на обучающую и тестовую выборки (4% данных — тестовая выборка)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.04, random_state=42)

# Преобразование обучающих и тестовых данных в векторное представление
X_tfidf = vectorizer.fit_transform(X_train)
test = vectorizer.transform(X_test)

# Инициализация и обучение модели PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(X_tfidf, y_train)

# Прогнозирование меток на тестовых данных
y_pred = pac.predict(test)

# Вычисление и вывод точности модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy * 100:.2f}%")

# Вывод отчета о классификации (precision, recall, f1-score)
print("Отчет о классификации:\n", classification_report(y_test, y_pred))

# Функция для отображения матрицы ошибок
def show_confusion_matrix(y_test, y_pred, class_names=['Marvel', 'DC']):
    confusion = sk_metrics.confusion_matrix(y_test, y_pred)
    confusion_normalized = confusion.astype('float') / confusion.sum(axis=1, keepdims=True)

    # Отображение матрицы ошибок с помощью heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_normalized, annot=True, cmap='Blues', fmt='.2f', xticklabels=class_names, yticklabels=class_names)

    plt.title("Матрица ошибок (Confusion Matrix)")
    plt.ylabel("Истинные метки")
    plt.xlabel("Предсказанные метки")

    plt.show()

# Вызов функции отображения матрицы ошибок
y_pred = pac.predict(test)
show_confusion_matrix(y_test, y_pred)

# График распределения классов в обучающей выборке
plt.figure(figsize=(6, 4))
sns.countplot(x=y_train, palette='Set2')
plt.title("Распределение классов в обучающей выборке")
plt.xlabel("Класс (0 = Marvel, 1 = DC)")
plt.ylabel("Количество")
plt.show()

# Гистограмма распределения рейтингов IMDB для фильмов Marvel и DC
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x='IMDB_Score', hue='Category', multiple='stack', kde=True, palette='Set1')
plt.title('Распределение рейтингов IMDB для Marvel и DC фильмов')
plt.xlabel('IMDB Рейтинг')
plt.ylabel('Количество фильмов')
plt.show()

# Гистограмма распределения годов выпуска фильмов Marvel и DC
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x='Year', hue='Category', multiple='stack', kde=False, palette='Set2')
plt.title('Годы выпуска фильмов для Marvel и DC')
plt.xlabel('Год выпуска')
plt.ylabel('Количество фильмов')
plt.show()

# Построение ROC-кривой для модели
y_prob = pac.decision_function(test)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Отображение ROC-кривой
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC-кривая (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend(loc='lower right')
plt.show()