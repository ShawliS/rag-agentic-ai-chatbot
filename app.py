import streamlit as st
from rag_graph import ask_question

st.set_page_config(page_title="Agentic AI eBook Chatbot", page_icon="🤖")
st.title("🤖 Agentic AI eBook Chatbot")
st.caption("Ask me anything about the Agentic AI eBook. I only answer using its content.")

if "history" not in st.session_state:
    st.session_state.history = []

# Re-display past messages on every rerun
for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        with st.expander(f"Retrieved context (confidence: {entry['confidence']})"):
            for i, chunk in enumerate(entry["chunks"], 1):
                st.markdown(f"**Chunk {i}:**\n\n{chunk}")

# Input box at the bottom of the page
question = st.chat_input("Type your question here...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the eBook..."):
            result = ask_question(question)
        st.write(result["answer"])
        with st.expander(f"Retrieved context (confidence: {result['confidence']})"):
            for i, chunk in enumerate(result["chunks"], 1):
                st.markdown(f"**Chunk {i}:**\n\n{chunk}")

    st.session_state.history.append({
        "question": question,
        "answer": result["answer"],
        "confidence": result["confidence"],
        "chunks": result["chunks"],
    })