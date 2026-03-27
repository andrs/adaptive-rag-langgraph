"""
AGENTIC TEXT-TO-SQL SYSTEM
--------------------------
This agent implements a ReAct (Reasoning + Acting) architecture to:
1. Translate natural language into ANSI SQL.
2. Dynamically explore database schemas.
3. Execute queries and return human-readable data insights.

Architecture: LangGraph + SQLAlchemy + OpenAI/VertexAI.
"""

import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# LangChain and LangGraph core components
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from config.credentials.credentials import PROJECT_ID
from config.credentials.credentials import LOCATION
from config.credentials.credentials import SERVICE_ACCOUNT

# Custom SQL manipulation tools from your local toolkit
from sql_toolkit import ( list_tables_tool, get_table_schema_tool, execute_sql_tool )

# Model Configuration Constants
MODEL_NAME_OPENAI = "gpt-4o-mini"
MODEL_NAME_VERTEXAI = "gemini-2.5-pro"

## 1. State Schema Definition
# We inherit from MessagesState to leverage automatic chat history management.
# 'user_query' is added to explicitly track the initial input independently from the message list.
class State(MessagesState):
    user_query: str

## 2. Graph Nodes (Logic Units)
def messages_builder(state: State):
    """
    Initialization Node:
    Constructs the System Prompt that governs the agent's persona and logic.
    Enforces a 4-step protocol: List Tables -> Read Schema -> Generate SQL -> Execute.
    """
    dba_sys_msg = (
        """
        You are a Sr. SQL Developer. Strictly follow this protocol:
        1. Identify relevant tables using 'list_tables_tool'.
        2. Retrieve the schema for those tables using 'get_table_schema_tool'.
        3. Generate an ANSI SQL query based on the schema and the user's question.
        4. Execute the query using 'execute_sql_tool' and provide the final data results.
        """
    )

    # Combine the system instructions with the current user input
    messages = [
        SystemMessage(dba_sys_msg),
        HumanMessage(state["user_query"])
    ]

    return {"messages": messages}

def dba_agent(state: State):
    """
    Agent Node (The Brain):
    Invokes the LLM with the current message history.
    The LLM decides whether to trigger a Tool Call or provide a final text response.
    """
    ai_message = dba_llm.invoke(state["messages"])
    # We name the message to track which node generated the response in the logs
    ai_message.name = "dba_agent"
    return {"messages": ai_message}

## 3. Control Flow Logic (Edges)
def should_continue(state: State):
    """
    Conditional Edge / Router:
    Determines the next path in the graph.
    If the LLM generated a 'tool_calls' payload, flow moves to the Execution node.
    Otherwise, the workflow terminates.
    """
    messages = state["messages"]
    last_message = messages[-1]

    # If the model requests a tool execution (e.g., listing tables or running SQL)
    if last_message.tool_calls:
        return "dba_tools"
    # If the model provides a final answer to the user
    return END



if __name__ == "__main__":
    print(f"--- Starting Text-to-SQL Intelligent Agent ---")
    load_dotenv()

    # LLM Provider Selection (VertexAI by default)
    #llm = ChatOpenAI( model=MODEL_NAME_OPENAI, temperature=0.0 )
    llm = ChatVertexAI(model_name=MODEL_NAME_VERTEXAI, temperature=0.0)

    ## 4. Tool Configuration
    # These tools allow the LLM to physically interact with the SQLite database.
    dba_tools = [list_tables_tool, get_table_schema_tool, execute_sql_tool]

    # Bind tools to the LLM so it understands 'how' and 'when' to call them
    dba_llm = llm.bind_tools(dba_tools, tool_choice="auto")

    ## 5. Workflow Construction
    workflow = StateGraph(State)

    # Adding Nodes to the Graph
    workflow.add_node("messages_builder", messages_builder)
    workflow.add_node("dba_agent", dba_agent)
    # ToolNode is a prebuilt helper that automatically executes the tools requested by the LLM
    workflow.add_node("dba_tools", ToolNode(dba_tools))

    # Defining Edges (Connections)
    workflow.add_edge(START, "messages_builder")
    workflow.add_edge("messages_builder", "dba_agent")

    # Cyclic Logic: dba_agent -> [tools or end]
    workflow.add_conditional_edges(
        source="dba_agent",
        path=should_continue,
        path_map={"dba_tools": "dba_tools", END: END}
    )

    # Crucial: After tool execution, the result returns to the agent for analysis
    workflow.add_edge("dba_tools", "dba_agent")

    # Compiling the Blueprint into a Runnable Graph
    react_graph = workflow.compile()

    ## 6. Database Setup (SQLite)
    # Use absolute paths to prevent 'Database Not Found' errors in different environments
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "sales.db")
    db_engine = create_engine(f"sqlite:///{db_path}")

    # Data Seeding (Resetting tables for a clean test run)
    with db_engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS sales"))
        conn.execute(text("CREATE TABLE sales (id INTEGER PRIMARY KEY, product TEXT, amount INTEGER)"))
        conn.execute(text("INSERT INTO sales (product, amount) VALUES ('Dell XPS 15', 10)"))

        conn.execute(text("DROP TABLE IF EXISTS products"))
        conn.execute(text("CREATE TABLE products (id INTEGER PRIMARY KEY, product TEXT)"))
        conn.execute(text("INSERT INTO products (product) VALUES ('Dell XPS 15')"))
        conn.commit()

    ## 7. Agent Execution
    # The 'config' dict injects the db_engine into the tools at runtime
    config = { "configurable": { "db_engine": db_engine } }
    inputs = { "user_query": "How many Dell XPS 15 laptops were sold?" }

    print(f"Question: {inputs['user_query']}\n")

    # Execute the graph synchronously
    final_state = react_graph.invoke(input=inputs, config=config)

    # Reasoning Process Visualization
    print("--- Agent Reasoning History ---")
    for m in final_state['messages']:
        m.pretty_print()