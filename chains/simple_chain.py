# generate query
# generate user response using query and schema
# manually run the query

from langchain_community.utilities import SQLDatabase
from langchain_community.chat_models import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel


db = SQLDatabase.from_uri("sqlite:///Chinook.db")


def get_schema(_):
    return db.get_table_info()


llm_name = "phi3"
llm = ChatOllama(model=llm_name)

sql_template = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:"""
sql_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Given an input question, convert it to a SQL query. No pre-amble."),
        ("human", sql_template),
    ]
)

sql_chain = (
    RunnablePassthrough.assign(
        schema=get_schema,
    )
    | sql_prompt
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)


ans_template = """Based on the table schema below, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""
ans_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given an input question and SQL response, convert it to a natural language answer. No pre-amble",
        ),
        ("human", ans_template),
    ]
)


class Input(BaseModel):
    question: str


final_chain = (
    RunnablePassthrough.assign(query=sql_chain).with_types(input_type=Input)
    | RunnablePassthrough.assign(
        schema=get_schema, response=lambda x: db.run(x["query"])
    )
    | ans_prompt
    | llm
)

final_chain.invoke({"question": "how many employees are there?"})
