import os
import shutil
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
import time
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

class PDFProcessor:
    def __init__(self, pdf_path, session_id):
        self.pdf_path = pdf_path
        self.session_id = session_id
        self.documents = PyPDFLoader(pdf_path).load()
        self.visuals = []
        self.tables = []
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vector_store = None
        self.split_docs = None
        self.img_dir = f".files/{session_id}/image_png"

    def extract_visuals_and_tables(self):
        pdf_document = fitz.open(self.pdf_path)
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
        self.vector_store = Chroma.from_documents(self.split_docs, self.embeddings, persist_directory=f"Chroma_db/{self.session_id}")

    def get_retriever(self, k=3):
        return self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})

class QueryHandler:
    def __init__(self, retriever, visuals, tables, pdf_path, pdf_name, session_id):
        self.retriever = retriever
        self.visuals = visuals
        self.tables = tables
        self.pdf_path = pdf_path  # Store full path
        self.pdf_name = pdf_name
        self.session_id = session_id
        self.llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.2)

        # Initialize session-specific conversation history if not already set
        if not cl.user_session.get("conversation_history"):
            cl.user_session.set("conversation_history", {})

        if session_id not in cl.user_session.get("conversation_history"):
            cl.user_session.get("conversation_history")[self.session_id] = []

    def add_to_history(self, query: str, answer: str):
        cl.user_session.get("conversation_history")[self.session_id].append((query, answer))
    
    def build_context(self) -> str:
        """Retrieve session-specific conversation history"""
        history = cl.user_session.get("conversation_history")[self.session_id]
        return "\n".join([f"Q: {q}\nA: {a}" for q, a in history]) if history else ""

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
        """Generate answer with session-based conversational memory and relevant document retrieval."""

        session_history = cl.user_session.get("conversation_history")[self.session_id]

        # Check if the query is a follow-up question
        is_follow_up = any(keyword in query.lower() for keyword in ["above", "previous", "last"])

        if is_follow_up and session_history:
            last_query, last_answer = session_history[-1]
            relevant_text = last_answer  # Use last response for context
        else:
            # Clear previous highlights in the PDF
            pdf_document = fitz.open(self.pdf_path)
            for page_number in range(len(pdf_document)):
                page = pdf_document[page_number]
                self.clear_highlights_on_page(page)
            pdf_document.saveIncr()
            pdf_document.close()

            # Retrieve relevant document excerpts
            docs = self.retriever.get_relevant_documents(query)
            relevant_text = "\n".join([doc.page_content for doc in docs])

            # Search for relevant tables
            relevant_tables = []
            for table in self.tables:
                for row in table['content']:
                    if any(query.lower() in cell.lower() for cell in row if cell):
                        relevant_tables.append(table)
                        break

            # Append relevant tables to relevant_text
            table_text = ""
            for table in relevant_tables:
                formatted_rows = [
                    " | ".join(str(cell) if cell else "" for cell in row)
                    for row in table['content']
                ]
                table_text += f"\nTable found on Page {table['page']}:\n" + "\n".join(formatted_rows)
            relevant_text += table_text

        instructions = (
            "You are an AI assistant tasked with answering questions based on the provided document. "
            "Your responses should be concise, accurate, and supported by the content from the document. "
            "Avoid making assumptions or adding information not present in the document."
        )

        context = self.build_context()
        prompt = (
            f"{instructions}\n\n {'Here is the conversation history to maintain context:\n' + context + '\n\n' if context else ''} \n Now, answer the following query:\n{query}\n\nRelevant content from the document or previous responses:\n{relevant_text}"
        )
        system_message = SystemMessage(content="You are an AI assistant helping to generate answers to questions based on the provided document.")
        user_message = HumanMessage(content=prompt)

        try:
            response = await self.llm.apredict_messages([system_message, user_message])
            answer = response.content.strip()

            # Handle citations for document sources
            citations = []
            if not is_follow_up:
                for doc in docs:
                    page_number = doc.metadata.get('page', None)
                    if page_number is not None:
                        self.highlight_text_in_pdf(page_number + 1, doc.page_content)
                        citations.append({"page": page_number + 1, "citation": f"From {self.pdf_name}, Page {page_number + 1}"})

                if not citations:
                    citations.append({"page": None, "citation": "Source page information not available"})

            # Store conversation history in session
            self.add_to_history(query, answer)

            return {"query": query, "answer": answer, "citations": citations}
        except Exception as e:
            return {"query": query, "answer": "Error in generating answer. Please check the API key, model access, and prompt clarity.", "citations": []}

@cl.on_chat_start
async def on_chat_start():
    session_id = cl.user_session.get('id')  # Generate unique session ID
    
    """Initialize session and PDF processing."""
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
    
    cl.user_session.set("pdf_path", file.path)
    cl.user_session.set("pdf_name", file.name)
    
    pdf_processor = PDFProcessor(file.path, cl.user_session.get("id"))
    pdf_processor.extract_visuals_and_tables()
    pdf_processor.process_documents()
    retriever = pdf_processor.get_retriever()

    cl.user_session.set("retriever", retriever)
    cl.user_session.set("visuals", pdf_processor.visuals)
    cl.user_session.set("tables", pdf_processor.tables)

    query_handler = QueryHandler(
        retriever=cl.user_session.get("retriever"),
        visuals=cl.user_session.get("visuals"),
        tables=cl.user_session.get("tables"),
        pdf_path=cl.user_session.get("pdf_path"),
        pdf_name=cl.user_session.get("pdf_name"),
        session_id=cl.user_session.get("id")
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


@cl.on_chat_end
async def on_chat_end():
    session_id = cl.user_session.get('id')
    
    file_dir = f".files/{session_id}"
    if os.path.exists(file_dir):
        shutil.rmtree(file_dir)