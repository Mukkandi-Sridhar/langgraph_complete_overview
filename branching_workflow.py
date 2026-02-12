from typing import TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class BranchState(TypedDict):
    input: str
    route: Literal["short", "long"]
    output: str

def router(state: BranchState):
    return {"route": "short" if len(state["input"]) < 40 else "long"}

def short_answer(state: BranchState):
    r = llm.invoke([HumanMessage(content=f"Short answer: {state['input']}")])
    return {"output": r.content}

def long_answer(state: BranchState):
    r = llm.invoke([HumanMessage(content=f"Detailed answer: {state['input']}")])
    return {"output": r.content}

graph = StateGraph(BranchState)
graph.add_node("router", router)
graph.add_node("short", short_answer)
graph.add_node("long", long_answer)
graph.set_entry_point("router")
graph.add_conditional_edges("router", lambda s: s["route"], {
    "short": "short",
    "long": "long"
})
graph.add_edge("short", END)
graph.add_edge("long", END)

agent = graph.compile()

if __name__ == "__main__":
    print(agent.invoke({"input": "Explain LangGraph"}))
