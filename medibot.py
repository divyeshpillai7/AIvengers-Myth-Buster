import os
import streamlit as st

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, HF_TOKEN):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )

def generate_response(prompt):
    CUSTOM_PROMPT_TEMPLATE = """
    Use the pieces of information provided in the context to answer user's question.
    If you don't know the answer, just say you don't know. Don't make anything up.

    Context: {context}
    Question: {question}

    Start the answer directly. No small talk please.
    """

    HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
    HF_TOKEN = os.environ.get("HF_TOKEN")

    vectorstore = get_vectorstore()
    if vectorstore is None:
        st.error("Vector store load failed.")
        return

    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )

    response = qa_chain.invoke({'query': prompt})
    result = response["result"]
    source_documents = response["source_documents"]
    formatted_sources = "\n\n".join([f"- {doc.metadata.get('source', 'Unknown')}" for doc in source_documents])
    return f"**Answer:** {result}\n\n---\n**Sources:**\n{formatted_sources}"

def main():
    st.set_page_config(page_title="AIvengers - Medical Myth Buster", page_icon="🦾", layout="wide")

    with st.sidebar:
        st.title("AIvengers")
        st.markdown("### About")
        st.info("Bust medical myths with AI! Powered by Mistral + LangChain.")

        st.markdown("### FAQ")
        with st.expander("❓ Can I ask anything?"):
            st.markdown("Yep, but answers are from uploaded docs only.")

        with st.expander("📊 How accurate is this?"):
            st.markdown("As accurate as your data!")

        

    # Custom CSS
    st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap');

    html, body, .stApp {
        background-color: #f2faff;
        font-family: 'Poppins', sans-serif;
        color: #000;
    }

    .stApp h1 {
        color: #007acc;
        text-align: center;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
    }

    .stApp h3, .stApp h4 {
        text-align: center;
        color: #333;
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
    }

    .stTextInput>div>div>input {
        background-color: #fff;
        border: 1px solid #a3d5ff;
        border-radius: 6px;
        padding: 10px;
        font-family: 'Poppins', sans-serif;
    }

    .st-emotion-cache-1yr0d2a, .st-emotion-cache-16txtl3 {
        background-color: #fff;
        border-radius: 12px;
        padding: 10px;
        margin: 10px 0;
        box-shadow: 0 3px 8px rgba(0,0,0,0.08);
        transition: transform 0.2s ease;
        color: #000 !important;
        font-family: 'Poppins', sans-serif;
    }

    .st-emotion-cache-1yr0d2a:hover, .st-emotion-cache-16txtl3:hover {
        transform: scale(1.01);
    }

    .stChatMessage {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        font-family: 'Poppins', sans-serif;
    }

    .stChatMessage[data-testid="stChatMessage"]:has(div[data-testid="stMarkdownContainer"]) {
        background-color: #007acc !important;
        border-left: 4px solid #e6f4ff;
        padding: 1rem;
        border-radius: 10px;
        color: #000 !important;
        margin-bottom: 1rem;
        font-family: 'Poppins', sans-serif;
    }

    /* Sidebar text color */
section[data-testid="stSidebar"] {
    color: white;
}

section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] .stMarkdown h4,
section[data-testid="stSidebar"] .stMarkdown h5,
section[data-testid="stSidebar"] .stMarkdown h6 {
    color: white !important;
}

section[data-testid="stSidebar"] {
    background-color: #1e1e2f; /* dark blue-gray */
}
    </style>
    """,
    unsafe_allow_html=True
)

    st.title("🧠 AIvengers - Medical Edition")
    st.subheader("Let’s Bust Some Medical Myths!")
    st.markdown("### Type your medical doubt below:")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Ask something medical...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        with st.spinner("Busting myths... 🧬"):
            try:
                result_to_show = generate_response(prompt)
                st.chat_message('assistant').markdown(result_to_show)
                st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})
                st.session_state.last_prompt = prompt
            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")

    if 'last_prompt' in st.session_state:
        if st.button("🔁 Regenerate Response"):
            st.chat_message('user').markdown(st.session_state.last_prompt)
            with st.spinner("Regenerating myth-busting... 🔄"):
                try:
                    result_to_show = generate_response(st.session_state.last_prompt)
                    st.chat_message('assistant').markdown(result_to_show)
                    st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})
                except Exception as e:
                    st.error(f"Something went wrong: {str(e)}")

if __name__ == "__main__":
    main()
