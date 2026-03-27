## 🤖 Text-to-SQL Intelligent Agent with LangGraph
An autonomous Sr. SQL Developer Agent capable of transforming natural language questions into precise ANSI SQL queries, executing them against a SQLite database, and returning human-readable insights.

Built with LangGraph to manage stateful, cyclic workflows and OpenAI/Google VertexAI for advanced reasoning.

## 🚀 Overview
This agent doesn't just "guess" SQL. It follows a strict ReAct (Reasoning + Acting) protocol to ensure accuracy and prevent hallucinations:

Exploration: Lists all available tables in the database.

Introspection: Inspects the schema (columns and types) of relevant tables.

Generation: Constructs a valid ANSI SQL query based on the retrieved metadata.

Execution: Runs the query and interprets the result for the user.

## 🛠️ Tech Stack
Orchestration: LangGraph

LLM: OpenAI gpt-4o-mini (or Google VertexAI gemini-2.5-pro)

Database: SQLite via SQLAlchemy

Environment: Python 3.10+

When asked: "How many Dell XPS 15 laptops were sold?", the agent performs the following steps:

1. Identify Tables
The agent calls list_tables_tool and discovers the products and sales tables.

2. Retrieve Schemas
It inspects both tables to understand the relationship (both contain a product column).

3. Generate & Execute SQL
The agent reasons that a JOIN is required and generates:

SQL
SELECT SUM(s.amount) AS total_sold 
FROM sales s 
JOIN products p ON s.product = p.product 
WHERE p.product = 'Dell XPS 15';
4. Final Result
Assistant: "A total of 10 Dell XPS 15 laptops were sold."

📝 Full Agent Logs
Plaintext
Question: How many Dell XPS 15 laptops were sold?

```
--- Agent Reasoning History ---

================================ System Message ================================

You are a Sr. SQL Developer. Strictly follow this protocol:
1. Identify relevant tables using 'list_tables_tool'.
2. Retrieve the schema for those tables using 'get_table_schema_tool'.
3. Generate an ANSI SQL query based on the schema and the user's question.
4. Execute the query using 'execute_sql_tool' and provide the final data results.

================================ Ai Message ==================================
Tool Calls: list_tables_tool (...) -> ["products", "sales"]

================================ Ai Message ==================================
Tool Calls: get_table_schema_tool (table_name: sales) -> [{'name': 'product', ...}, {'name': 'amount', ...}]

================================ Ai Message ==================================
Tool Calls: execute_sql_tool (query: SELECT SUM(s.amount) FROM sales ...) -> [(10,)]

================================ Ai Message ==================================
```
> Result
> 
> A total of 10 Dell XPS 15 laptops were sold.

🔧 Setup
Clone the repository.

Install dependencies: pip install langgraph langchain-openai sqlalchemy python-dotenv.

Add your OPENAI_API_KEY to the .env file.

Run the agent: python sql_agent.py.# adaptive-rag-langgraph