# Import necessary modules and define env variables

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os
import io
import chainlit as cl
import PyPDF2
from io import BytesIO


from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Chainlit automatically loads the api keys.
# OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")


# text_splitter and system template

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

Example of your response should be:

```
The answer is foo
SOURCES: xyz
```

Begin!
----------------
{summaries}"""


messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}


@cl.on_chat_start
async def on_chat_start():

    # Sending an image with the local file path

    await cl.Message(content="Hello there, Welcome to AskAnyQuery related to Data!",).send()
    files = None
    print("[DEBUG] File None")
    # Wait for the user to upload a PDF file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!",
            accept=["application/pdf"],
            max_size_mb=3,
            timeout=180,
        ).send()

    file = files[0]
    print("[DEBUG] File Type: ",type(file))
    print("[DEBUG] File Name: ", file.name)
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Read the PDF file
    print("[DEBUG] Streaming PDF")
    with open(file.path, "rb") as f:  # Open in read-binary mode
        pdf_content = f.read()
    pdf_stream = BytesIO(pdf_content)

    print("[DEBUG] Converting into PDF")
    pdf = PyPDF2.PdfReader(pdf_stream)
    print("[DEBUG] Extracting text")
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()
    print("[DEBUG] PDF Text Extract:", len(pdf_text)) 
    # Split the text into chunks
    print("[DEBUG] Splitting into chunks")
    texts = text_splitter.split_text(pdf_text)

    # Create metadata for each chunk
    print("[DEBUG] Making Metadata")
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a Chroma vector store
    print("[DEBUG] Running Embeddings")
    embeddings = OpenAIEmbeddings(disallowed_special=())
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )

    # Create a chain that uses the Chroma vector store
    print("[DEBUG] Running chain. ")
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
    )

    # Save the metadata and texts in the user session
    print("[DEBUG] Saving metadata and texts. ")
    cl.user_session.set("metadatas", metadatas)
    cl.user_session.set("texts", texts)

    # Let the user know that the system is ready
    print("[DEBUG] System ready. ")
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: str):
    print("[DEBUG] Initial message DEBUG:", type(message))
    chain = cl.user_session.get("chain")  # type: RetrievalQAWithSourcesChain
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    print("[DEBUG] Before passing to chain-type:", type(message))
    res = await chain.invoke(message, callbacks=[cb])
    print("[DEBUG] Passed to chain type:", type(message),type(res))
    answer = res["answer"]
    sources = res["sources"].strip()
    source_elements = []
    
    # Get the metadata and texts from the user session
    metadatas = cl.user_session.get("metadatas")
    all_sources = [m["source"] for m in metadatas]
    texts = cl.user_session.get("texts")

    if sources:
        found_sources = []

        # Add the sources to the message
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            # Get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = texts[index]
            found_sources.append(source_name)
            # Create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
    else:
        await cl.Message(content=answer, elements=source_elements).send()