import logging
import os
import sys

from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from setup.config import OPENAI_API_KEY

logging.getLogger("pypdf").setLevel(logging.ERROR)

# ---------- Step 1: Load files ------------------------
loader = PyPDFDirectoryLoader("files/")
documents = loader.load()

# ---------- Step 2: Split into chunks ---------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)
docs = text_splitter.split_documents(documents)

# ---------- Step 3: Create vectorstore -------------
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
FAISS_INDEX_PATH = "faiss_index"
if os.path.exists(FAISS_INDEX_PATH):
    print("Loading existing vectorstore from disk...")
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    print("Building vectorstore for first time...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)  # ← saves to disk
    print(f"Vectorstore saved to '{FAISS_INDEX_PATH}' folder")

# ---------- Step 4: QA Chain -----------------------
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True
)

# ---------- Step 5: Terminal Loop ------------------
print("\n📄 Files loaded and ready!")
print("Type your question below. Type 'exit' to quit.\n")

while True:
    query = input("Question: ").strip()
    if query.lower() == "exit":
        break
    if not query:  # ← handles empty enter
        print("Please enter a question.\n")
        continue

    answer = qa_chain.invoke(query)

    print("\n" + "-" * 50)
    print("ANSWER:")
    print(answer['result'])

    print("\nSOURCES:")
    for i, doc in enumerate(answer['source_documents']):
        page = doc.metadata.get('page', 'N/A')
        page = page + 1 if isinstance(page, int) else page
        print(f"[Chunk {i + 1}] Page {page} \n {doc.page_content}")
    print("-" * 50 + "\n")

# Note : FAISS does similarity search, not page order search. It returns chunks ranked by relevance to your query — not by their position in the document:
