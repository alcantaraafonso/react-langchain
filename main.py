from typing import List, Union
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain_core.agents import AgentAction, AgentFinish
from langchain.tools import Tool, tool
from langchain.tools.render import render_text_description

from callbacks import AgentCallbackHandler


load_dotenv()

# @tool decorator is used to define a function as a tool to langchain
@tool
def get_text_length(text: str) -> int:
    """
    returns the length of the text
    """ 
    print(f"Calculating length of text: {text}")

    text = text.strip("'\n").strip('"')
    return len(text)

def find_tool_by_name(tools: List[Tool], tool_name) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")

if __name__ == '__main__':
    # print("Hello, World!")
    tools = [get_text_length] # list of tools -> passa a instancia de get_text_length

    template = """
        Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought: {agent_scratchpad}
        """
    
    # {agent_scratchpad} is a special variable that is used to store the state of the agent
    # recieves the inputs from the user
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names= ", ".join([tool.name for tool in tools]),
        )
    
    llm = ChatOpenAI(temperature=0, stop = ["\nObservation", "Observation"], callbacks=[AgentCallbackHandler()])
    intermediate_steps = []

    agent = (
        {
            "input": lambda x:x["input"], 
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])
        } 
        | prompt 
        | llm 
        | ReActSingleInputOutputParser()
    ) #it`s a LCEL pipeline

    agent_step = ""
    while not isinstance(agent_step, AgentFinish):
        agent_step = agent.invoke(
            {
                "input": "What is the length of the word dog",
                "agent_scratchpad": intermediate_steps,
            }
        )

        print(f"{agent_step=}")
        
        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input

            observation = tool_to_use.func(str(tool_input))

            print(f"{observation=}")
            intermediate_steps.append((agent_step, str(observation))) #append the action and the observation

    print(f"{agent_step=}")
    
    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input

        observation = tool_to_use.func(str(tool_input))

        print(f"{observation=}")
        intermediate_steps.append((agent_step, str(observation)))

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length of the word: DOG",
            "agent_scratchpad": intermediate_steps,
        }
    )
    
    print(agent_step)
    if isinstance(agent_step, AgentFinish):
        print("### AgentFinish ###")
        print(agent_step.return_values)        
