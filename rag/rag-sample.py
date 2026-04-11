import os
import sys

from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from setup.config import OPENAI_API_KEY

# ---------- Step1 : Load data ------------------------
loader = TextLoader("data.txt")
documents = loader.load()

# ---------- Step2 : configure data splitter & create chunks ------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=30,
    separators=["\n", ". ", " "]
)
docs = text_splitter.split_documents(documents)

# ---------- Step3 : create vectorstore ----------------------
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_documents(docs, embeddings)

# ---------- Step4 : User LLM to get the final answer providing vectorstore ----------------------
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2})
)

# RUN
query = "does paris weather unchanged?"
answer = qa_chain.invoke(query)
print(answer['result'])

# ------- DUMMY Ignore -------------
# print(f"Total chunks: {len(docs)}\n")
# for i, doc in enumerate(docs):
#     print(f"Chunk {i}: {doc.page_content}")


# print(f"Vectorstore created with {(vectorstore.index)} vectors")
#
# retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
# retrieved_docs = retriever.invoke(query)
#
# for i, doc in enumerate(retrieved_docs):
#     print(f"\nChunk {i}:")
#     print(doc.page_content)
