from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class LoopState(TypedDict):
    input: str
    count: int
    output: str

def improve(state: LoopState):
    r = llm.invoke([HumanMessage(content=f"Improve text: {state['input']}")])
    return {"output": r.content, "count": state["count"] + 1}

def stop(state: LoopState):
    return END if state["count"] >= 2 else "improve"

graph = StateGraph(LoopState)
graph.add_node("improve", improve)
graph.set_entry_point("improve")
graph.add_conditional_edges("improve", stop)

agent = graph.compile()

if __name__ == "__main__":
    print(agent.invoke({"input": "LangGraph is powerful", "count": 0}))
