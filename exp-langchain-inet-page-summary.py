import questionary as q
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.document_loaders.web_base import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains.llm import LLMChain
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain_core.documents import Document

MODEL = "dolphin-mixtral"
# MODEL = "llama3:latest"
TEMPERATURE = 0.8
VERBOSE = True
TEMPLATES = {
    "dolphin-mixtral": """
        <|im_start|>system
        You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you asked anything about real-time information or information from the Internet - assume that it is retrived for you and is in the Context.
        If you don't know the answer, just say that you don't know.
        <|im_end|>
        <|im_start|>user
        Question: {question}
        Context: {context}
        <|im_end|>
        <|im_start|>assistant
    """,
    "llama3": """
        You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you asked anything about real-time information or information from the Internet - assume that it is retrived for you and is in the Context.
        If you don't know the answer, just say that you don't know.
        Question: {question}
        Context: {context}
        The answer:
    """,
}


def load_website_content(url):
    docs: list[Document] = WebBaseLoader(url).load()
    print(f"Loaded {len(docs)} docs...")
    return docs[0].page_content


def create_retrieval_qa_chain(db: Chroma):
    QA_CHAIN_PROMPT_TEMPLATE = TEMPLATES[MODEL]

    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=QA_CHAIN_PROMPT_TEMPLATE,
    )

    return RetrievalQA.from_chain_type(
        llm=Ollama(
            model=MODEL,
            temperature=TEMPERATURE,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        ),
        retriever=db.as_retriever(search_kwargs={"k": 1}),
        chain_type_kwargs={"prompt": prompt},
        verbose=VERBOSE,
    )


def main():
    url = q.select(
        "What site do you want to pull the latest hottest headlines from?",
        ["https://news.ycombinator.com/", "your option"],
    ).ask()
    url = q.text("Enter the site URL").ask() if url == "your option" else url
    content = load_website_content(url)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    texts = text_splitter.create_documents([content])

    db = Chroma.from_documents(texts, OllamaEmbeddings(model=MODEL))
    chain = create_retrieval_qa_chain(db)
    question = f"""
        What are the top (10 maximum) latest and the most popular
        (based on number of points and comments if such information's available)
        headlines that you can derive from the given context which is a content of a web-page?"""
    chain.invoke(question)

    # just some empty space for the output readability
    for i in range(3):
        print()


if __name__ == "__main__":
    main()
