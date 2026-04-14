import logging
import os
import sys

from langchain_classic.chains.retrieval_qa.base import RetrievalQA

logging.getLogger("pypdf").setLevel(logging.ERROR)

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from setup.config import OPENAI_API_KEY

# ---------- Step 1: Load PDF ------------------------
loader = PyPDFLoader("Anthropic_Certification_Terms_and_Conditions.pdf")
documents = loader.load()

# ---------- Step 2: Split into chunks ---------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)
docs = text_splitter.split_documents(documents)
docs = [doc for doc in docs if len(doc.page_content.strip()) > 30]

# ---------- Step 3: Create vectorstore -------------
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_documents(docs, embeddings)

# ---------- Step 4: QA Chain -----------------------
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True
)

# ---------- Step 5: Terminal Loop ------------------
print("\n📄 PDF loaded and ready!")
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
