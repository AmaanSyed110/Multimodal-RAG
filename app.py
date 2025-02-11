import streamlit as st
import pdfplumber
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os
from PIL import Image, ImageOps, UnidentifiedImageError
import io
import pytesseract  # Tesseract OCR
import fitz  # PyMuPDF
import imageio
import logging
import threading

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate environment variables
if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API key.")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize the embedding model with OpenAI
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-3-large"
)

# Initialize the LLM for conversational retrieval
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4o",
    temperature=0.7
)

def extract_images_from_pdf(pdf_docs):
    """Extract all images from uploaded PDF documents using PyMuPDF."""
    images = []
    try:
        for pdf in pdf_docs:
            pdf_file = fitz.open(stream=pdf.read(), filetype="pdf")
            for page_number in range(len(pdf_file)):
                page = pdf_file.load_page(page_number)
                image_list = page.get_images(full=True)
                logging.info(f"Found {len(image_list)} images on page {page_number + 1}.")
                for img_index, img in enumerate(image_list):
                    xref = img[0]  # XREF of the image
                    base_image = pdf_file.extract_image(xref)
                    image_data = base_image["image"]
                    try:
                        # Attempt to open the image using Pillow
                        img_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
                        images.append(img_pil)
                        logging.info(f"Successfully extracted image {img_index + 1} from page {page_number + 1}.")
                    except Exception as e:
                        logging.warning(f"Attempting to convert image {img_index + 1} on page {page_number + 1}...")
                        try:
                            # Convert unsupported formats using imageio
                            img_array = imageio.imread(io.BytesIO(image_data))
                            img_pil = Image.fromarray(img_array).convert("RGB")
                            images.append(img_pil)
                            logging.info(f"Successfully converted and extracted image {img_index + 1} from page {page_number + 1}.")
                        except Exception as e:
                            logging.error(f"Skipped image {img_index + 1} on page {page_number + 1} due to an error: {str(e)}")
    except Exception as e:
        logging.error(f"Error extracting images from PDF: {e}")
        st.error("Failed to extract images from the PDF. Please check the file format.")
    return images

def preprocess_image(image):
    """Preprocess image for better OCR accuracy."""
    try:
        # Convert to grayscale
        image = ImageOps.grayscale(image)
        # Resize image
        image = image.resize((800, 800))  # Adjust size as needed
        return image
    except Exception as e:
        logging.error(f"Error preprocessing image: {e}")
        return image

def extract_text_from_images(images):
    """Extract text from images using Tesseract OCR."""
    ocr_text = ""
    try:
        for img in images:
            img = preprocess_image(img)  # Preprocess image
            text = pytesseract.image_to_string(img)
            ocr_text += text + "\n"
            logging.info(f"Extracted text from an image of size {img.size}.")
    except Exception as e:
        logging.error(f"Error extracting text from image: {e}")
    return ocr_text

def get_pdf_text_and_tables(pdf_docs):
    """Extract text and tables from uploaded PDF documents."""
    text = ""
    tables = []
    try:
        for pdf in pdf_docs:
            with pdfplumber.open(pdf) as pdf_reader:
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""  # Extract text
                    tables.extend(page.extract_tables())  # Extract tables
    except Exception as e:
        logging.error(f"Error extracting text and tables from PDF: {e}")
        st.error("Failed to extract text and tables from the PDF. Please check the file format.")
    return text, tables

def get_text_chunks(text):
    """Split text into smaller chunks for processing."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1500,  # Set chunk size to 1500 characters for optimal performance
        chunk_overlap=300,  # Overlap of 300 characters for better context retention
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks, tables):
    """Create a FAISS vector store from text chunks and tables."""
    try:
        # Embed text chunks
        text_embeddings = embeddings.embed_documents(text_chunks)
        logging.info(f"Generated {len(text_embeddings)} text embeddings.")
        
        # Embed tables (convert tables to text and embed)
        table_texts = ["Table: " + "\n".join(["\t".join(map(str, row)) for row in table]) for table in tables]
        table_embeddings = embeddings.embed_documents(table_texts)
        logging.info(f"Generated {len(table_embeddings)} table embeddings.")
        
        # Combine all embeddings
        combined_embeddings = text_embeddings + table_embeddings
        combined_texts = text_chunks + table_texts
        
        # Create FAISS vector store
        vectorstore = FAISS.from_embeddings(list(zip(combined_texts, combined_embeddings)), embeddings)
        logging.info("FAISS vector store created successfully.")
        return vectorstore
    except Exception as e:
        logging.error(f"Error creating vector store: {e}")
        st.error("Failed to create vector store. Please check the input data.")
        return None

def get_conversation_chain(vectorstore):
    """Create a conversation chain for the chatbot."""
    try:
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model="gpt-4o",
            temperature=0.7
        )
        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain
    except Exception as e:
        logging.error(f"Error creating conversation chain: {e}")
        st.error("Failed to create conversation chain. Please check the OpenAI API key.")
        return None
    
def process_pdf(pdf):
    """Process a single PDF and return its text, tables, and images."""
    try:
        raw_text, tables = get_pdf_text_and_tables([pdf])
        images = extract_images_from_pdf([pdf])
        return raw_text, tables, images
    except Exception as e:
        logging.error(f"Error processing PDF: {e}")
        return "", [], []


def handle_user_input(user_question):
    """Process user input and get response using the conversation chain."""
    if st.session_state.conversation:
        try:
            # Handle general queries using the conversation chain
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']
            
            # Display the updated chat history
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(f"**User:** {message.content}")
                else:
                    st.write(f"**Assistant:** {message.content}")
            
            # Debug: Print retrieved documents
            if "source_documents" in response:
                for doc in response["source_documents"]:
                    st.write(f"Retrieved document: {doc.page_content}")
        except Exception as e:
            logging.error(f"Error handling user input: {e}")
            st.error("An error occurred while processing your request. Please try again.")

def main():
    """Main application function."""
    st.set_page_config(page_title="Chat with Multi-Modal RAG", page_icon=":books:")
    st.title("Chat with Your Multi-Modal RAG :books:")

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = {"text": "", "tables": [], "images": []}
    if "combined_text" not in st.session_state:
        st.session_state.combined_text = ""

    # User question input
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_user_input(user_question)

    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.subheader("Upload your PDFs")
        pdf_docs = st.file_uploader(
            "Upload PDFs here:",
            accept_multiple_files=True,
            type="pdf"
        )

        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing your documents..."):
                    try:
                        # Initialize variables to store combined data
                        all_text = ""
                        all_tables = []
                        all_images = []
                        
                        # Process each PDF in parallel using threading
                        threads = []
                        results = []
                        for pdf in pdf_docs:
                            thread = threading.Thread(target=lambda p=pdf: results.append(process_pdf(p)))
                            threads.append(thread)
                            thread.start()
                        for thread in threads:
                            thread.join()
                        
                        # Combine results
                        for result in results:
                            raw_text, tables, images = result
                            all_text += raw_text + "\n"
                            all_tables.extend(tables)
                            all_images.extend(images)
                        
                        # Extract text from images using OCR
                        ocr_text = extract_text_from_images(all_images)
                        
                        # Combine OCR text with PDF text
                        combined_text = all_text + "\n" + ocr_text
                        
                        if combined_text.strip() == "" and not all_tables:
                            st.error("No readable content found in the uploaded PDFs. Please check the PDFs.")
                        else:
                            # Process text chunks
                            text_chunks = get_text_chunks(combined_text)
                            
                            # Create vector store
                            vectorstore = get_vectorstore(text_chunks, all_tables)
                            
                            # Update session state with processed data
                            st.session_state.processed_data = {
                                "text": combined_text,
                                "tables": all_tables,
                                "images": all_images
                            }
                            st.session_state.combined_text = combined_text
                            
                            # Create conversation chain
                            st.session_state.conversation = get_conversation_chain(vectorstore)
                            st.success("Documents processed successfully! You can now ask questions.")
                    except Exception as e:
                        logging.error(f"Error processing documents: {e}")
                        st.error("An error occurred while processing your documents. Please check the files and try again.")
            else:
                st.warning("Please upload at least one PDF to process.")

if __name__ == "__main__":
    main()