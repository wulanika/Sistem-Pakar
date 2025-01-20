from flask import Flask, render_template, request, jsonify, session
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
# Import untuk Random Forest
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# --- Load Random Forest model dan TF-IDF vectorizer ---
filename_model = 'random_forest_model_kana.sav'
filename_vectorizer = 'tfidf_vectorizer_kana.sav'

loaded_model = pickle.load(open(filename_model, 'rb'))
loaded_vectorizer = pickle.load(open(filename_vectorizer, 'rb'))

# --- Fungsi untuk preprocess teks ---
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('indonesian'))
    tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

# --- Fungsi untuk prediksi masalah ---
def predict_problem(user_input):
    # Preprocess teks input pengguna
    preprocessed_text = preprocess_text(user_input) 
    
    # Transform teks menggunakan vectorizer yang sudah dimuat
    text_vec = loaded_vectorizer.transform([preprocessed_text])
    
    # Prediksi menggunakan model Random Forest
    prediction = loaded_model.predict(text_vec)[0]
    
    return prediction

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Memuat vectorstore yang sudah ada dari direktori "data"
try:
    vectorstore = Chroma(
        persist_directory="data",
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/bert-base-nli-max-tokens")
    )
    print("Vectorstore berhasil dimuat.")
except Exception as e:
    print(f"Kesalahan saat memuat vectorstore: {e}")
    vectorstore = None

if vectorstore:
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
else:
    retriever = None

# Menyiapkan model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, max_tokens=None, timeout=None)

# Membuat memori
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Membuat prompt dengan memori
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
        "Anda adalah asisten tentang penyakit buah jagung."
        "Anda hanya memproses jawaban berdasarkan data pada dataset"
        "Jika Anda tidak tahu jawabannya, katakan bahwa Anda tidak tahu. Gunakan maksimal tiga kalimat dan buat jawaban yang singkat."
        "Sistem ini bernama CornCare."
        "Dataset ini kami ambil dari seorang yang bernama Ibu Anelce Tipimbu, S.P.  Yang bekerja di Dinas Pertahanan Pangan dan dan Pertanian kabupaten Raja Ampat, menjabat sebagai kepala seksi perlindungan tanaman."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)

# Membuat chain dengan LLMChain
conversation_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/index")
def konsultasi():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_message = data.get('query', '').strip()
    if not user_message:
        return jsonify({"error": "Pesan tidak boleh kosong"}), 400

    # Prediksi menggunakan Random Forest (opsional, jika ingin tetap menggunakannya)
    prediction = predict_problem(user_message)

    # Ambil atau inisialisasi riwayat percakapan
    session.setdefault("chat_history", [])

    result = conversation_chain.run({"question": user_message})

    # Simpan riwayat percakapan
    session["chat_history"].append({"sender": "user", "message": user_message})
    session["chat_history"].append({"sender": "bot", "message": result})

    return jsonify({"answer": result})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    session.pop("chat_history", None)
    memory.clear()
    return jsonify({"status": "success", "message": "Riwayat percakapan telah dihapus"})

if __name__ == '__main__':
    app.run(debug=True)