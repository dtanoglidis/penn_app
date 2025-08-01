import os
import streamlit as st
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage


# Large Language Model
llm = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0)

# Embeddings Model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

from langchain_core.vectorstores import InMemoryVectorStore

# Initialize vector store
vector_store = InMemoryVectorStore(embeddings)

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


syllabi_loader = PyPDFDirectoryLoader(path='data/syllabi')
syllabi_docs = syllabi_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=200)
all_splits_syllabi = text_splitter.split_documents(syllabi_docs)

_ = vector_store.add_documents(documents=all_splits_syllabi)

from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool

graph_builder = StateGraph(MessagesState)

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=4)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode

# ======================================================================
# Define elements to check if the question is relevant 

def is_about_penn_course(user_query: str) -> bool:
    system_prompt = (
        "You are a classifier. Only answer 'yes' or 'no'.\n"
        "Is the user's question about courses, syllabi, or departments at the University of Pennsylvania?\n\n"
        f"User: {user_query}\nAnswer:"
    )
    response = llm.invoke([HumanMessage(content=system_prompt)])
    return "yes" in response.content.strip().lower()

def classify_query(state: MessagesState):
    """Classify user input as in-domain (course-related) or not."""
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            user_msg = msg.content
            break
    else:
        return {"is_valid": False}  # fail-safe

    return {"is_valid": is_about_penn_course(user_msg)}

def reject_off_topic(state: MessagesState):
    """Politely reject questions that are not about Penn classes."""
    return {
        "messages": [
            AIMessage(content="I'm sorry, I can only help with questions about Penn courses or syllabi.")
        ]
    }
# ======================================================================
# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}

# Step 2: Execute the retrieval.
tools_node = ToolNode([retrieve])

# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = system_message_content = (
    "You are an AI assistant for the University of Pennsylvania. "
    "Ask the student/person about their department or school, if you don't know it, "
    "You can only answer questions related to Penn classes, departments, or syllabi. "
    "Before responding ask yourself: is what being asked or what I am going to say relavant to Penn courses or syllabi?"
    "If yes respond. If not, **you should say: 'I am sorry, I can only help with questions about Penn courses or syllabi.'**\n\n "
    "Respond only for topics you have information in your content and do not hallucinate"
    "Follow these rules strictly:\n"
    "- ❗ Always ask the user's school/department first if you don't know it.\n"
    "- ❗ Never answer unrelated questions — instead, politely refuse.\n"
    "- ❗ Do not invent answers or speculate about courses you don’t have info on.\n"
    "- ✅ Use only the retrieved documents (context) below to answer.\n\n"
    " The classes you have access to are the following:"
    "- ASTR006: The Solar System, Exoplanets, and Life"
    "- BIOL/PHYS 5566: Machine Learning Methods In Natural Science Modeling"
    "- Introduction to AI: Concepts, Applications, and Impact"
    "- PHYS 3359: Data Analysis for the Natural Sciences II: Machine Learning"
    "- PHYS6632: QUANTUM FIELD THEORY II"
    f"{docs_content}"
)
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}

from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition

graph_builder.add_node("query_or_respond",query_or_respond)
#graph_builder.add_node("classify_query",classify_query)
#graph_builder.add_node("reject_off_topic",reject_off_topic)
graph_builder.add_node("tools",tools_node)
graph_builder.add_node("generate",generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)
# If tool call detected, classify it first
#graph_builder.add_conditional_edges(
#    "query_or_respond",
#    tools_condition,
#    {
#        END: END,  # No tool call, just respond
#        "classify_query": "classify_query",  # Proceed to classifier if tool needed
#    },
#)

# If valid, continue to tools; else reject
#graph_builder.add_conditional_edges(
#    "classify_query",
#    lambda x: "tools" if x["is_valid"] else "reject_off_topic",
#    {
#        "tools": "tools",
#        "reject_off_topic": "reject_off_topic"
#    }
#)

#graph_builder.add_edge("tools", "generate")
#graph_builder.add_edge("generate", END)
#graph_builder.add_edge("reject_off_topic", END)

graph = graph_builder.compile()

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

