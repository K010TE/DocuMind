import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# --- CONFIGURAÇÃO INICIAL ---
st.set_page_config(page_title="DocuMind", page_icon="🧠")

# Recupera variáveis de ambiente (vindas do arquivo .env via Docker)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_URL = f"http://{QDRANT_HOST}:6333"

st.title("🧠 DocuMind: Docker + Groq")
st.caption("Arquitetura: Embeddings Locais (CPU) + LLM na Nuvem (Groq)")

# --- VERIFICAÇÃO DE SEGURANÇA ---
if not GROQ_API_KEY:
    st.error("🚨 ERRO CRÍTICO: Chave da API Groq não encontrada.")
    st.info("Certifique-se de ter criado o arquivo .env e configurado o docker-compose.yaml corretamente.")
    st.stop()

# --- INICIALIZAÇÃO DOS MODELOS (CACHEADO) ---
# Usamos @st.cache_resource para não recarregar os modelos a cada clique
@st.cache_resource
def get_embeddings():
    # Modelo leve que roda bem na CPU do container
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_llm():
    # Atualizado para o modelo mais recente (Llama 3.3)
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, api_key=GROQ_API_KEY)

embeddings = get_embeddings()
llm = get_llm()

# --- BARRA LATERAL: INGESTÃO DE DADOS ---
with st.sidebar:
    st.header("📂 Área de Upload")
    uploaded_file = st.file_uploader("Solte seu PDF aqui", type="pdf")
    
    if uploaded_file and st.button("Processar Documento"):
        with st.spinner("Lendo e Vetorizando... (Isso roda na CPU local)"):
            try:
                # 1. Cria arquivo temporário para o PyPDFLoader ler
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                # 2. Carrega o texto
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                
                # 3. Fatia o texto (Chunking)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(docs)
                
                # 4. Salva no Banco Vetorial (Qdrant)
                QdrantVectorStore.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    url=QDRANT_URL,
                    collection_name="documind_collection",
                    force_recreate=True # Substitui o documento anterior
                )
                
                st.success("✅ Processamento Concluído! Pode perguntar.")
                
            except Exception as e:
                st.error(f"Erro ao processar: {e}")
            finally:
                # Limpeza do arquivo temporário
                if 'tmp_path' in locals():
                    os.remove(tmp_path)

# --- ÁREA PRINCIPAL: CHAT (RAG) ---
st.header("💬 Chat com IA")

pergunta = st.chat_input("Pergunte algo sobre o conteúdo do PDF...")

if pergunta:
    # 1. Mostra a pergunta do usuário
    with st.chat_message("user"):
        st.write(pergunta)

    # 2. Processa a resposta
    with st.chat_message("assistant"):
        try:
            # Conecta ao Qdrant para buscar referências
            client = QdrantClient(url=QDRANT_URL)
            vectorstore = QdrantVectorStore(
                client=client, 
                collection_name="documind_collection", 
                embedding=embeddings
            )
            
            # Busca os 3 trechos mais parecidos com a pergunta
            docs = vectorstore.similarity_search(pergunta, k=3)
            
            if not docs:
                st.warning("⚠️ Não encontrei informações relevantes no PDF. Tente reformular.")
                st.stop()
                
            # Monta o contexto (cola os trechos encontrados)
            contexto_texto = "\n\n".join([d.page_content for d in docs])
            
            # Monta o Prompt para a IA
            prompt = f"""
            Você é um assistente especialista. Use APENAS o contexto abaixo para responder a pergunta.
            Se a resposta não estiver no contexto, diga que não sabe. Não invente informações.
            
            CONTEXTO:
            {contexto_texto}
            
            PERGUNTA: 
            {pergunta}
            """
            
            # Gera a resposta via Streaming (efeito de digitação)
            stream = llm.stream(prompt)
            st.write_stream(stream)
            
            # (Opcional) Mostra o que a IA leu para responder
            with st.expander("Ver contexto utilizado"):
                st.info(contexto_texto)
                
        except Exception as e:
            st.error(f"Erro na comunicação: {e}")
            st.info("Verifique se o container do Qdrant está rodando e se sua chave Groq é válida.")