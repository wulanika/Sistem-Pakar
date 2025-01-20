import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pickle  # Impor library pickle

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

# Baca dataset
df = pd.read_csv('anisa.csv', sep=';', encoding='latin-1')  # Ganti 'DATASET.csv' dengan nama file datasetmu

# Fungsi untuk preprocess teks
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('indonesian'))  # Ganti 'indonesian' jika bahasa dataset berbeda
    tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()  # Ganti dengan Sastrawi jika perlu
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

df['Teks'] = df['Teks'].apply(preprocess_text)

# Bagi data 
X_train, X_test, y_train, y_test = train_test_split(
    df['Teks'], df['Label'], test_size=0.2, random_state=42
)

# TfidfVectorizer 
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.8)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --- Oversampling dengan SMOTE ---
oversampler = SMOTE(random_state=42, k_neighbors=2)  # Gunakan k_neighbors
X_train_vec_resampled, y_train_resampled = oversampler.fit_resample(X_train_vec, y_train)

# --- Tuning Hyperparameter Random Forest dengan GridSearchCV ---
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'max_features': ['sqrt', 'log2'],  # Hapus 'auto'
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train_vec_resampled, y_train_resampled)
best_rf_model = grid_search.best_estimator_

# --- Prediksi dan Evaluasi ---
y_pred_rf = best_rf_model.predict(X_test_vec)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Akurasi model Random Forest: {accuracy_rf}')

# --- Confusion Matrix ---
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
plt.imshow(cm_rf, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Random Forest)')
plt.colorbar()
tick_marks = range(len(set(y_test))) 
plt.xticks(tick_marks, set(y_test))
plt.yticks(tick_marks, set(y_test))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print(classification_report(y_test, y_pred_rf))

# --- Simpan model dan vectorizer ---
filename_model = 'random_forest_model.sav'
pickle.dump(best_rf_model, open(filename_model, 'wb'))

filename_vectorizer = 'tfidf_vectorizer.sav'
pickle.dump(vectorizer, open(filename_vectorizer, 'wb'))