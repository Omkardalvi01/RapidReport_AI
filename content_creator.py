from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from typing import Optional
from langchain_community.utilities.twilio import TwilioAPIWrapper
from langgraph.graph import StateGraph, END
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.agents import Tool
import os
from langchain.agents import initialize_agent, AgentType
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()
endpoint = FastAPI()

#API keys
API_KEY = os.getenv("GEMINI_KEY")
ACCOUNT_SID = os.getenv("account_sid")
TWILIO_AUTH_TOKEN = os.getenv("auth_token")
PHONE_NO = os.getenv("PHONE_NO")

# Setup tools
web_search = DuckDuckGoSearchRun()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
twilio = TwilioAPIWrapper(
    account_sid=ACCOUNT_SID,
    auth_token=TWILIO_AUTH_TOKEN,
    from_number="whatsapp:+14155238886"
)
tools = [
    Tool(name='websearch', func=web_search.run, description="Tool used to search web"),
    Tool(name='wiki_search', func=wikipedia.run, description="Tool used to wikipedia search which is a free online encyclopedia")
]


class State(BaseModel):
    query: str 
    keyword: Optional[list[str]] = None
    info: Optional[str] = None
    good: Optional[str] = None
    bad: Optional[str] = None
    report: Optional[str] = None



llm = ChatGoogleGenerativeAI(model='models/gemini-2.0-flash-lite', api_key=API_KEY)

def keywords(state: State) -> State:
    output_parser = CommaSeparatedListOutputParser()
    formal_instr = output_parser.get_format_instructions()
    prompt = PromptTemplate(
        template="List the topic of query and 5 relevant keywords to the topic {query}.\n{format_instructions}", 
        input_variables=['query'], 
        partial_variables={'format_instructions': formal_instr}
    )
    chain = prompt | llm | output_parser
    output = chain.invoke(input={'query': state.query})
    return State(
        query=state.query,
        keyword=output,
        info=state.info,
        good=state.good,
        bad=state.bad,
        report=state.report
    )

def info(state: State) -> State:
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)
    answer = agent.invoke({"input": f'Find information on {state.query} and {state.keyword}'})['output']
    return State(
        query=state.query,
        keyword=state.keyword,
        info=answer,
        good=state.good,
        bad=state.bad,
        report=state.report
    )

def good_impacts(state: State) -> State:
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)
    answer = agent.invoke({"input": f'Find all the good impacts of topic in this query: {state.query}'})['output']
    return State(
        query=state.query,
        keyword=state.keyword,
        info=state.info,
        good=answer,
        bad=state.bad,
        report=state.report
    )

def bad_impacts(state: State) -> State:
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)
    answer = agent.invoke({"input": f'Find all the bad impacts of topic in this query: {state.query}'})['output']
    return State(
        query=state.query,
        keyword=state.keyword,
        info=state.info,
        good=state.good,
        bad=answer,
        report=state.report
    )

def report(state: State) -> State:
    prompt = PromptTemplate(
        template="Take everything from {info}, as well as good impacts such as {good} and bad impacts such as {bad} and create a detailed report for it" \
        "Generate only the HTML content, without explanations or comments. Return just the HTML to be rendered in a browser." \
        "Maake it long",
        input_variables=['info', 'good', 'bad']
    )
    chain = prompt | llm
    output = chain.invoke({'info': state.info, 'good': state.good, 'bad': state.bad})
    return State(
        query=state.query,
        keyword=state.keyword,
        info=state.info,
        good=state.good,
        bad=state.bad,
        report=output.content
    )

def snd_report(state: State) -> State:
    twilio.run(str(state.keyword), PHONE_NO)
    return State(
        query=state.query,
        keyword=state.keyword,
        info=state.info,
        good=state.good,
        bad=state.bad,
        report=state.report
    )


workflow = StateGraph(State)
workflow.add_node('keywords', keywords)
workflow.add_node('information', info)
workflow.add_node('good_stuff', good_impacts)
workflow.add_node('bad_stuff', bad_impacts)
workflow.add_node('ovr_report', report)
workflow.add_node('send_report', snd_report)


workflow.set_entry_point('keywords')
workflow.add_edge('keywords', 'information')
workflow.add_edge('information', 'good_stuff')
workflow.add_edge('good_stuff', 'bad_stuff')
workflow.add_edge('bad_stuff', 'ovr_report')
workflow.add_edge('ovr_report', 'send_report')
workflow.add_edge('send_report', END)

app = workflow.compile()

class QueryInput(BaseModel):
        query: str
    
@endpoint.post("/")
def submit_query(body: QueryInput):
    state_input = {"query": body.query}
    result = app.invoke(state_input)
    # with open('Report.md', 'w+', encoding='utf-8') as f:
    #     f.write(result['report'])
    return JSONResponse(content={"response" : result['report']})

from fastapi.middleware.cors import CORSMiddleware

endpoint.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(endpoint, host="127.0.0.1", port=8000)