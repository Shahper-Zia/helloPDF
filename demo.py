import os
import warnings
from typing import List, Tuple
import chainlit as cl
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import HumanMessage, SystemMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import pdfplumber
import fitz  # PyMuPDF
import time  # new

warnings.filterwarnings("ignore")

# Set Google API Key
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = ""

class PDFProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.documents = PyPDFLoader(pdf_path).load()
        self.visuals = []
        self.tables = []
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vector_store = None
        self.split_docs = None

    def extract_visuals_and_tables(self):
        pdf_document = fitz.open(self.pdf_path)
        self.img_dir = "image_png"
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        # Extract images
        for page_number in range(len(pdf_document)):
            page = pdf_document[page_number]
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                img_path = os.path.join(self.img_dir, f"page-{page_number + 1}_image-{img_index + 1}.png")
                with open(img_path, "wb") as f:
                    f.write(image_bytes)
                self.visuals.append({
                    "type": "image",
                    "path": img_path,
                    "page": page_number + 1,
                    "description": f"Figure {img_index + 1}"
                })

        # Extract tables
        with pdfplumber.open(self.pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                tables_on_page = page.extract_tables()
                for table_index, table in enumerate(tables_on_page):
                    if table:
                        self.tables.append({"type": "table", "content": table, "page": i + 1})

        pdf_document.close()

    def process_documents(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.split_docs = text_splitter.split_documents(self.documents)
        self.vector_store = Chroma.from_documents(self.split_docs, self.embeddings)

    def get_retriever(self, k=3):
        return self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})

class QueryHandler:
    def __init__(self, retriever, visuals, tables, pdf_path, pdf_name):
        self.retriever = retriever
        self.visuals = visuals
        self.tables = tables
        self.pdf_path = pdf_path  # Store full path
        self.pdf_name = pdf_name
        self.llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.2)
        self.conversation_history = []

    def add_to_history(self, query: str, answer: str):
        self.conversation_history.append((query, answer))

    def build_context(self) -> str:
        context = "\n".join([f"Q: {q}\nA: {a}" for q, a in self.conversation_history])
        return context

    def clear_highlights_on_page(self, page):
        # CHANGE THIS LINE
        pdf_document = fitz.open(self.pdf_path)  # Changed from pdf_name to pdf_path
        annotations = page.annots()
        if annotations:
            for annot in annotations:
                if annot.type[0] == 8:  # Highlight annotation type
                    page.delete_annot(annot)
        pdf_document.saveIncr()
        pdf_document.close()

    def highlight_text_in_pdf(self, page_number: int, text: str):
        pdf_document = fitz.open(self.pdf_path)
        page = pdf_document[page_number - 1]  # Pages are 0-indexed in PyMuPDF

        self.clear_highlights_on_page(page)
        text_instances = page.search_for(text)
        for inst in text_instances:
            highlight = page.add_highlight_annot(inst)
            highlight.update()

        pdf_document.saveIncr()
        pdf_document.close()

    async def generate_answer(self, query: str) -> dict:
        # Check if the query is a follow-up question
        is_follow_up = any(keyword in query.lower() for keyword in ["above", "previous", "last"])

        if is_follow_up and self.conversation_history:
            # Use the last answer from the conversation history
            last_query, last_answer = self.conversation_history[-1]
            relevant_text = last_answer
        else:
            pdf_document = fitz.open(self.pdf_path)  # ← THIS IS THE CRITICAL FIX
            for page_number in range(len(pdf_document)):
                page = pdf_document[page_number]
                self.clear_highlights_on_page(page)
            pdf_document.saveIncr()
            pdf_document.close()

            docs = self.retriever.get_relevant_documents(query)
            relevant_text = "\n".join([doc.page_content for doc in docs])

            # Search for relevant tables
            relevant_tables = []
            for table in self.tables:
                for row in table['content']:
                    if any(query.lower() in cell.lower() for cell in row if cell):
                        relevant_tables.append(table)
                        break

            # Append relevant tables to the relevant_text
            table_text = ""
            for table in relevant_tables:
                table_text += f"\nTable found on Page {table['page']}:\n" + "\n".join([" | ".join(row) for row in table['content']])
            relevant_text += table_text

        instructions = (
            "You are an AI assistant tasked with answering questions based on the provided document. "
            "Your responses should be concise, accurate, and supported by the content from the document. "
            "Avoid making assumptions or adding information not present in the document."
        )

        context = self.build_context()
        if context:
            prompt = (
                f"{instructions}\n\n"
                f"Here is the conversation history to help you understand the context:\n"
                f"{context}\n\n"
                f"Now, answer the following query:\n{query}\n\n"
                f"Based on the following relevant text from the document or previous answers:\n{relevant_text}"
            )
        else:
            prompt = (
                f"{instructions}\n\n"
                f"Answer the following query:\n{query}\n\n"
                f"Based on the following relevant text from the document:\n{relevant_text}"
            )

        system_message = SystemMessage(content="You are an AI assistant helping to generate answers to questions based on the provided document.")
        user_message = HumanMessage(content=prompt)

        try:
            response = await self.llm.apredict_messages([system_message, user_message])
            answer = response.content.strip()

            citations = []
            if not is_follow_up:
                for doc in docs:
                    page_number = doc.metadata.get('page', None)
                    if page_number is not None:
                        self.highlight_text_in_pdf(page_number + 1, doc.page_content)
                        citations.append({"page": page_number + 1, "citation": f"From {self.pdf_name}, Page {page_number + 1}"})

                if not citations:
                    citations.append({"page": None, "citation": "Source page information not available"})

            self.add_to_history(query, answer)

            return {"query": query, "answer": answer, "citations": citations}
        except Exception as e:
            return {"query": query, "answer": "Error in generating answer. Please check the API key, model access, and prompt clarity.", "citations": []}


@cl.on_chat_start
async def on_chat_start():
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]
    msg = cl.Message(content=f"Processing {file.name}...")
    await msg.send()

    # Store both path and name in session
    cl.user_session.set("pdf_path", file.path)
    cl.user_session.set("pdf_name", file.name)

    pdf_processor = PDFProcessor(file.path)
    pdf_processor.extract_visuals_and_tables()
    pdf_processor.process_documents()
    retriever = pdf_processor.get_retriever()

    query_handler = QueryHandler(
        retriever, 
        pdf_processor.visuals, 
        pdf_processor.tables, 
        pdf_path=file.path,  # Pass full path instead of just name
        pdf_name=file.name
    )

    cl.user_session.set("query_handler", query_handler)
    msg.content = f"Processing {file.name} done. You can now ask questions!"
    await msg.update()


@cl.on_message
async def main(message: cl.Message):
    query_handler = cl.user_session.get("query_handler")
    response = await query_handler.generate_answer(message.content)
    answer = response["answer"]
    citations = response["citations"]

    unique_citations = {}
    for citation in citations:
        page = citation["page"]
        if page not in unique_citations:
            unique_citations[page] = citation

    citation_texts = "\n\nCitations:\n" + "\n".join([c["citation"] for c in unique_citations.values()])

    elements = []
    for citation in unique_citations.values():
        if citation["page"] is not None:
            elements.append(cl.Pdf(
                name=f"{query_handler.pdf_name}",
                display="side",
                path=query_handler.pdf_path,
                page=citation["page"],
                link_text=citation["citation"]
            ))

    await cl.Message(content=f"{answer}{citation_texts}", elements=elements).send()

def clean_image_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

@cl.on_chat_end
async def on_chat_end():
    pdf_path = cl.user_session.get("pdf_path")
    if pdf_path and os.path.exists(pdf_path):
        try:
            os.remove(pdf_path)
        except Exception as e:
            print(f"Error cleaning up file: {str(e)}")

    # Clean up the image_png folder
    img_dir = "image_png"
    if os.path.exists(img_dir):
        clean_image_folder(img_dir)