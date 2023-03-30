
from langchain.agents import initialize_agent, load_tools
from langchain.llms import OpenAI, OpenAIChat
from langchain.chat_models import ChatOpenAI
from aiohttp import ClientSession
from langchain.chains.conversation.memory import ConversationBufferMemory

FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here MUST translates to chinese]
```"""

class LangChainChat():
    def __init__(self, engine="gpt-3.5-turbo", verbose=False):
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.verbose = verbose
        self.engine = engine

    async def aask(self, q):
        aiosession = ClientSession()
        llm = ChatOpenAI(temperature=0, max_tokens=2000, model_name=self.engine)
        async_tools = load_tools(["serpapi"], llm=llm, aiosession=aiosession)
        async_agent = initialize_agent(async_tools, 
                                       llm, 
                                       agent="conversational-react-description", 
                                       memory=self.memory, 
                                       verbose=self.verbose,
                                       agent_kwargs =dict(
                                            format_instructions = FORMAT_INSTRUCTIONS
                                       ))
        rsp =  await async_agent.arun(q)
        await aiosession.close()
        return rsp
    
    def clear(self):
        self.memory.clear()

    def ask(self, q):
        llm = OpenAI(temperature=0, max_tokens=2000, model_name=self.engine)
        tools = load_tools(["serpapi"], llm=llm)
        agent = initialize_agent(
            tools, llm, agent="conversational-react-description", verbose=True, memory=self.memory,
            agent_kwargs =dict(
                format_instructions = FORMAT_INSTRUCTIONS
            ))
        
        return dict(content=agent.run(input=q))

if __name__=="__main__":
    # import asyncio
    import sys
    import os
    os.environ["OPENAI_API_KEY"] = "x"
    os.environ["SERPAPI_API_KEY"] ="x"
    # chat = LangChainChat(verbose=True)
    # asyncio.run(chat.ask(sys.argv[1]))


