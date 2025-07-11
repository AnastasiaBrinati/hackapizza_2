import pandas as pd
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. Carica .env e la chiave
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("⚠️ La variabile OPENAI_API_KEY non è stata trovata nel file .env")

os.environ["OPENAI_API_KEY"] = api_key

# 2. Carica tutti i menu (PDF)
loader = DirectoryLoader(
    "Hackapizza Dataset/Menu",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
documents = loader.load()

# 3. Dividi i testi in chunk
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# 4. Crea embeddings
embedding = OpenAIEmbeddings()

# 5. Costruisci FAISS
db = FAISS.from_documents(docs, embedding)

# 6. Crea retriever
retriever = db.as_retriever(search_kwargs={"k": 5})

# Prompt personalizzato con 'context' come variabile documenti
prompt_template = """Sei un assistente che risponde a domande su menu di ristoranti.
Utilizza esclusivamente queste informazioni estratte:

{context}

Rispondi alla domanda in modo chiaro e conciso:
Domanda: {question}
Risposta:"""

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

# 7. Crea LLM con modello GPT-3.5-turbo
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 8. Catena RetrievalQA con prompt personalizzato
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# 9. Carica domande dal CSV
df_domande = pd.read_csv("Hackapizza Dataset/domande.csv")

# 10. Itera sulle domande e rispondi mostrando risposta e documenti usati
for idx, row in df_domande.iterrows():
    query = row["domanda"]
    print(f"\n➡️ Domanda {idx + 1}: {query}")
    result = qa_chain.invoke(query)
    print("✅ Risposta:", result['result'])
    print("📄 Documenti usati come fonte:")
    """
    for doc in result['source_documents']:
        source = doc.metadata.get("source", "sconosciuto")
        content_preview = doc.page_content[:200].replace("\n", " ")  # primi 200 caratteri puliti
        print(f"------ Fonte: {source}\n  Contenuto estratto: {content_preview}...\n")
    """
    if idx == 3:
        break