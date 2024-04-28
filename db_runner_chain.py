# generate query
# generate user response using query and schema
# use query runner to run the query

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.chat_models import ChatOllama
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from operator import itemgetter

from sql_prompts import PROMPTS

# todo: use different llms to generate query and ans
llm_name = "phi3"
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
llm = ChatOllama(model=llm_name)


def _strip(text: str) -> str:
    return text.strip()


def create_query_chain(llm, db: SQLDatabase):
    # select a prompt based on the db dialect, or else use a default prompt
    prompt_to_use = PROMPTS[db.dialect]

    inputs = {
        "input": lambda x: x["question"] + "\nSQLQuery: ",
        "table_info": lambda x: db.get_table_info(
            table_names=x.get("table_names_to_use")
        ),
    }

    return (
        RunnablePassthrough.assign(**inputs)
        | (
            lambda x: {
                k: v
                for k, v in x.items()
                if k not in ["question", "table_names_to_use"]
            }
        )
        | prompt_to_use.partial(top_k=10)
        | llm.bind(stop=["\nSQLResult"])
        | StrOutputParser()
        | _strip()
    )


# chain to generate sql query
query_chain = create_query_chain(llm, db)

# tool to execute sql query
execute_tool = QuerySQLDataBaseTool(db=db)

# create answer chain
ans_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.
    
    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer: """
)
ans_chain = ans_prompt | llm | StrOutputParser()

final_chain = (
    RunnablePassthrough.assign(query=query_chain).assign(
        result=itemgetter("query") | execute_tool
    )
    | ans_chain
)
