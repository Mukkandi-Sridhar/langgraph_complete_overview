from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class SeqState(TypedDict):
    input: str
    output: str

def step1(state: SeqState):
    r = llm.invoke([HumanMessage(content=f"Summarize: {state['input']}")])
    return {"output": r.content}

def step2(state: SeqState):
    r = llm.invoke([HumanMessage(content=f"Rewrite clearly: {state['output']}")])
    return {"output": r.content}

graph = StateGraph(SeqState)
graph.add_node("step1", step1)
graph.add_node("step2", step2)
graph.set_entry_point("step1")
graph.add_edge("step1", "step2")
graph.add_edge("step2", END)

agent = graph.compile()

if __name__ == "__main__":
    print(agent.invoke({"input": "LangGraph enables structured workflows"}))
