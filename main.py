from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_parse import LlamaParse
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from dotenv import load_dotenv
import streamlit as st
import os

#carregando o arquivo .env
load_dotenv()

#salvando as chaves
openai_key = os.environ["OPENAI_API_KEY"]
llama_cloud_key  = os.environ["LLAMA_CLOUD_API_KEY"]

#instanciando o llm da OpenAI
llm = OpenAI(model="gpt-3.5-turbo",api_key=openai_key,timeout=30.0)

#instanciando o objeto LlamaParse
parser = LlamaParse(result_type="markdown",api_key=llama_cloud_key,language="pt")
file_extractor = {".txt":parser}
Settings.embed_model = OpenAIEmbedding()
embed_model = OpenAIEmbedding(model="text-embedding-3-small",embed_batch_size=50)

#carregando os dados do diretorio data
documents = SimpleDirectoryReader("./data",file_extractor=file_extractor).load_data()
vector_index = VectorStoreIndex.from_documents(documents=documents,embed_model=embed_model)
unicamp_engine = vector_index.as_query_engine(llm=llm)

#Inicializacao da aplicação streamlit
st.title("Chatbot vestibular Unicamp 2025")
st.write("Faça perguntas sobre o vestibular da Unicamp")

#salvando a pergunta do usuario em uma string
pergunta = st.text_input("Qual a sua dúvida?")

if st.button("Enviar"):
    if pergunta:
        resposta = unicamp_engine.query(pergunta)
        st.write("Resposta")
        st.write(resposta.response)
    else:
        st.write("Por favor digite uma pergunta:")