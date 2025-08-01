import streamlit as st
from langgraph.graph import END
from rag_bot import graph
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
import os 

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# Large Language Model
llm = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0)


def is_penn_course_question(user_input: str) -> bool:
    """Use the LLM to decide if the query is about Penn classes/syllabi."""
    system_prompt = (
        "You are a classification assistant. Your task is to decide whether a user's question "
        "is about courses, syllabi, or departments at the University of Pennsylvania (Penn). "
        "Respond only with 'yes' or 'no'.\n\n"
        "Example 1:\nUser: What's a good elective for a Wharton student?\nAnswer: yes\n\n"
        "Example 2:\nUser: How do I make banana bread?\nAnswer: no\n\n"
        "*Note: simple greetings or reasonable follow ups should count as yes"
        f"User: {user_input}\nAnswer:"
    )
    response = llm.invoke([HumanMessage(content=system_prompt)])
    return "yes" in response.content.lower()


# Setup persistent memory
THREAD_ID = "abc123"
CONFIG = {"configurable": {"thread_id": THREAD_ID}}

#  Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "graph" not in st.session_state:
    st.session_state.graph = graph

# Page config
st.set_page_config(page_title="Penn RAG Chatbot", page_icon="data/images/Penn_logo.png")
#st.title("Penn SyllaBOT")
#st.image("data/images/Penn_logo.png", width=120)


with st.sidebar:
    st.image("data/images/Penn_logo_2.png", width=200)
    st.markdown("## Penn SyllaBOT")
    st.caption("Course Q&A assistant - Ask your questions!")


left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image("data/images/Penn_logo.png", width=500)

#st.markdown(
#    "<h1 style='text-align: center;'>Penn SyllaBOT</h1>",
#    unsafe_allow_html=True
#)

st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='font-size: 36px;'>Penn SyllaBOT</h1>
        <p style='font-size: 18px; color: gray; margin-top: -10px;'>
            Ask questions about available AI courses :)
        </p>
        <hr style='margin-top: 20px; margin-bottom: 30px; border: none; border-top: 2px solid #ccc;' />
    </div>
    """,
    unsafe_allow_html=True
)

# Chat input
user_input = st.chat_input("Ask about courses, your major, etc...")

# Display previous messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 📤 Handle new input
#if user_input:
#    # Display user message immediately
#    st.chat_message("user").markdown(user_input)
#    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Stream graph execution
#    with st.chat_message("assistant"):
#        with st.spinner("Thinking..."):
#            full_response = ""
#            for step in st.session_state.graph.stream(
#                {"messages": [{"role": "user", "content": user_input}]},
#                stream_mode="values",
#                config=CONFIG,
#            ):
#                last_msg = step["messages"][-1]

                # Only print actual AI/Tool response
#                if last_msg.type == "ai":
#                    new_text = last_msg.content[len(full_response):]
#                    full_response += new_text
#                    st.write(new_text, unsafe_allow_html=True)

            # Save assistant message
#            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
if user_input:
    # Display user message immediately
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if is_penn_course_question(user_input):
                full_response = ""
                for step in st.session_state.graph.stream(
                    {"messages": [{"role": "user", "content": user_input}]},
                    stream_mode="values",
                    config=CONFIG,
                ):
                    last_msg = step["messages"][-1]

                    if last_msg.type == "ai":
                        new_text = last_msg.content[len(full_response):]
                        full_response += new_text
                        st.write(new_text, unsafe_allow_html=True)

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": full_response}
                )
            else:
                rejection = "I'm sorry, I can only help with questions about Penn courses or syllabi."
                st.markdown(rejection)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": rejection}
                )
