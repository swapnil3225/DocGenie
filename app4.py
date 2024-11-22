import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from htmlTemplates import css, bot_template, user_template
import google.generativeai as genai
from deep_translator import GoogleTranslator
import time
import logging

# Initialize logging
logging.basicConfig(filename='app.log', level=logging.ERROR)

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Helper functions
def validate_pdf_files(pdf_docs):
    valid_files = []
    invalid_files = []
    for pdf in pdf_docs:
        if pdf.type != "application/pdf":
            invalid_files.append(pdf.name)
        else:
            valid_files.append(pdf)
    return valid_files, invalid_files

def get_pdf_text(pdf_docs):
    text = ""
    total_pages = 0
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        total_pages += len(pdf_reader.pages)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text, total_pages

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversation_chain(vectorstore):
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question, selected_language):
    # Check if conversation is initialized
    if st.session_state.conversation is None:
        st.error("Please process the documents first to start the conversation.")
        return

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            translated_message = translate_response(message.content, selected_language)
            st.write(bot_template.replace("{{MSG}}", translated_message), unsafe_allow_html=True)

    st.markdown("""
        <script>
            const chatBox = document.querySelector('.stVerticalBlockBorderWrapper');
            if (chatBox) {
                chatBox.scrollTop = chatBox.scrollHeight;
            }
        </script>
    """, unsafe_allow_html=True)


def translate_response(text, target_language):
    if target_language == "English":
        return text  # No translation needed

    language_codes = {
        "Marathi": "mr",
        "Hindi": "hi",
        "Bengali": "bn",
        "Gujarati": "gu",
        "Punjabi": "pa",
        "Tamil": "ta",
        "Telugu": "te",
        "Malayalam": "ml",
        "Kannada": "kn",
        "Odia": "or"
    }

    if target_language in language_codes:
        try:
            return GoogleTranslator(source='en', target=language_codes[target_language]).translate(text)
        except Exception as e:
            return f"Translation failed: {e}"
    else:
        return "Language not supported"

def reset_if_idle():
    if "last_interaction" in st.session_state:
        if time.time() - st.session_state.last_interaction > 300:  # 5 minutes timeout
            st.session_state.chat_history = []
            st.session_state.conversation = None
    st.session_state.last_interaction = time.time()

def save_chat_history():
    chat_history_text = "\n".join([f"User: {msg.content}" if i % 2 == 0 else f"Bot: {msg.content}"
                                  for i, msg in enumerate(st.session_state.chat_history)])
    return chat_history_text


# Main function
def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    reset_if_idle()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with multiple PDFs :books:")


    # Sidebar for uploading documents and selecting language
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

        valid_pdf_docs, invalid_pdf_docs = validate_pdf_files(pdf_docs)
        if invalid_pdf_docs:
            st.error(f"The following files are not PDFs: {', '.join(invalid_pdf_docs)}")

        if st.session_state.conversation is None:
            selected_language = st.selectbox(
                "Select language for response",
                ["English", "Marathi", "Hindi", "Bengali", "Gujarati", "Punjabi", "Tamil", "Telugu", "Malayalam", "Kannada", "Odia"]
            )
        else:
            # Disable language selection if conversation is already active
            selected_language = st.selectbox(
                "Select language for response",
                ["English", "Marathi", "Hindi", "Bengali", "Gujarati", "Punjabi", "Tamil", "Telugu", "Malayalam", "Kannada", "Odia"],
                disabled=True
            )

        if st.button("Process"):
            if valid_pdf_docs:
                with st.spinner("Processing your documents..."):
                    raw_text, total_pages = get_pdf_text(valid_pdf_docs)
                    st.success(f"Successfully processed {total_pages} pages.")
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)

        # Button to clear the chat and reset everything
        if st.button("Clear Chat"):
            st.session_state.conversation = None
            st.session_state.chat_history = []
            st.rerun()  # This will reload the app to reset everything

        # Button to download chat history
        if st.session_state.chat_history:
            chat_history = save_chat_history()
            st.download_button("Download Chat History", chat_history, file_name="chat_history.txt")

    # Use custom CSS to place the input box at the bottom
    st.markdown("""
        <style>
            .stTextInput {
                position: fixed;
                bottom : -2px;
                margin: 0 auto;
                width: 50%;
                left: 35%;
                z-index: 10;
                padding-bottom:40px;
                background-color : rgb(14, 17, 23);
            }
            .stTextInput > div > input {
                background-color: #333;
                color: #fff;
            }
        </style>
    """, unsafe_allow_html=True)

    # Query input at the bottom of the page
    user_question = st.text_input("Ask a Question about your documents:")
    if user_question:
        handle_userinput(user_question, selected_language)

if __name__ == '__main__':
    main()
