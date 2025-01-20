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
filename_model = 'random_forest_model.sav'
filename_vectorizer = 'tfidf_vectorizer.sav'

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
    
    # Return prediksi tanpa mencetaknya ke log
    return prediction

load_dotenv()

app = Flask(__name__)  # Perbaikan nama variabel

# Set secret key untuk manajemen sesi
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
            "Anda adalah asisten aplikasi B-Care, sebuah platform diagnosa gejala dini kanker payudara. Anda bertugas memberikan solusi terkait pencegahan dan penanganan gejala dini kanker payudara."
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/coba')
def konsultasi():
    return render_template('coba.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/about_us')
def about_us():
    return render_template('about_us.html')

@app.route('/get', methods=['GET'])
def get_response():
    user_message = request.args.get('msg', '').strip()  # Validasi input
    if not user_message:
        return jsonify({"error": "Pesan tidak boleh kosong"}), 400
    
    # Prediksi menggunakan Random Forest
    prediction = predict_problem(user_message)
    
    # Ambil atau inisialisasi riwayat percakapan
    session.setdefault("chat_history", [])
    
    # Proses input dengan chain tanpa menambahkan prediksi ke dalam prompt
    result = conversation_chain.run({"question": user_message})
    
    # Simpan riwayat percakapan
    session["chat_history"].append({"sender": "user", "message": user_message})
    session["chat_history"].append({"sender": "bot", "message": result})
    
    return jsonify(result)  # Kembalikan respons ke front-end

@app.route('/clear_history', methods=['POST'])
def clear_history():
    session.pop("chat_history", None)
    memory.clear()
    return jsonify({"status": "success", "message": "Riwayat percakapan telah dihapus"})

if __name__ == '__main__':
    app.run(debug=True)