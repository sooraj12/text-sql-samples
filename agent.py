from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.chat_models import ChatOllama
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)

from langchain.agents.output_parsers.tools import ToolsAgentOutputParser
from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langchain.agents.agent import AgentExecutor, RunnableMultiActionAgent

from typing import cast
from agent_prompts import SQL_AGENT_PREFIX, SQL_FUNCTIONS_SUFFIX

llm_name = "phi3"
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
toolkit_llm = ChatOllama(model=llm_name)
agent_llm = ChatOllama(model=llm_name)

# todo: customize agent with custom prompts (few-shot examples)


def create_tool_calling_agent(llm, tools, prompt):
    llm_with_tools = llm.bind_tools(tools)

    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_tool_messages(x["intermediate_steps"])
        )
        | prompt
        | llm_with_tools
        | ToolsAgentOutputParser()
    )

    return agent


def create_sql_agent():
    extra_tools = []

    # create toolkit for the agent to use
    toolkit = SQLDatabaseToolkit(llm=toolkit_llm, db=db)
    tools = toolkit.get_tools() + list(extra_tools)

    # create prompt
    prefix = SQL_AGENT_PREFIX
    prefix.format(dialect=toolkit.dialect, top_k=10)

    messages = [
        SystemMessage(content=cast(str, prefix)),
        HumanMessagePromptTemplate.from_template("{input}"),
        AIMessage(content=SQL_FUNCTIONS_SUFFIX),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    # create agent runnable
    runnable = create_tool_calling_agent(agent_llm, tools, prompt)

    agent = RunnableMultiActionAgent(
        runnable=runnable, input_keys_arg=["input"], return_keys_arg=["output"]
    )

    return AgentExecutor(
        name="SQL Agent Executor",
        agent=agent,
        tools=tools,
        max_iterations=15,
        verbose=True,
    )


agent_executor = create_sql_agent()

agent_executor.invoke({"input": "Describe playlisttrack table"})
