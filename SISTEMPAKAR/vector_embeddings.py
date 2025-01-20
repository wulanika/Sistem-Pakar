from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import os

# Memuat variabel lingkungan dari file .env
load_dotenv()
folder_path = "source_data"  # Ganti dengan path folder

# Mengambil semua file PDF dalam folder
pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]

# Inisialisasi list untuk menyimpan data dari semua PDF
all_data = []

for pdf_file in pdf_files:
    try:
        # Membaca setiap PDF
        file_path = os.path.join(folder_path, pdf_file)
        loader = PyPDFLoader(file_path)
        data = loader.load()
        all_data.extend(data)
        print(f"Berhasil memuat {len(data)} halaman dari {pdf_file}")
    except Exception as e:
        print(f"Kesalahan saat memuat PDF {pdf_file}: {e}")

if all_data:
    # Membagi dokumen menjadi bagian kecil
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(all_data)
    print(f"Jumlah total bagian dokumen: {len(docs)}")
else:
    print("Tidak ada data untuk diproses.")

# Menginisialisasi embeddings dan penyimpanan vektor
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/bert-base-nli-max-tokens")
    
    # Membuat dan menyimpan vektor ke dalam penyimpanan Chroma di direktori 'data'
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="data"  # Menyimpan penyimpanan vektor di direktori 'data'
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    print("Penyimpanan vektor berhasil dibuat dan disimpan.")
except Exception as e:
    print(f"Kesalahan saat menginisialisasi embeddings atau penyimpanan vektor: {e}")

# Menginisialisasi LLM dan membuat RAG chain
try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, max_tokens=None, timeout=None)
    
    # Menyiapkan memori
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Membuat prompt dengan memori
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
),
            MessagesPlaceholder(variable_name="chat_history"),  # Memori akan otomatis disisipkan di sini
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )

    # Membuat chain dengan memori
    conversation_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory,
        verbose=True
    )

    # Mengambil query dari pengguna
    query = "Apa itu penyakit jagung?"  # Ganti dengan pertanyaan yang diinginkan
    
    # Memproses query untuk mendapatkan respons dari LLM
    response = conversation_chain.run(question=query)  # Memproses pertanyaan dengan chain yang sudah terintegrasi dengan memori
    print("Respons:", response)  # Menampilkan respons dari model

    # Mengambil dokumen relevan menggunakan retriever berdasarkan similarity
    retrieved_docs = retriever.invoke(query)
    
    # Menghitung similarity untuk setiap dokumen yang diambil dengan respons
    similarities = []
    query_embedding = embeddings.embed_query(response)  # Mendapatkan embedding untuk jawaban LLM
    
    for doc in retrieved_docs:
        doc_embedding = embeddings.embed_query(doc.page_content)  # Mendapatkan embedding untuk dokumen
        similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]  # Menghitung cosine similarity
        similarities.append((doc.page_content, similarity))
    
    # Mengurutkan hasil berdasarkan similarity
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    
    print("Hasil pencarian berdasarkan respons dengan tingkat kemiripan:")
    for content, score in similarities:
        print(f"Tingkat kemiripan: {score:.4f}")

except Exception as e:
    print(f"Kesalahan saat menginisialisasi LLM dan memori: {e}")
