
---

## 🤖 `react_agent.py`

```python
import os
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class AgentState(TypedDict):
    question: str
    reasoning: str
    answer: str

def reason(state: AgentState):
    r = llm.invoke([HumanMessage(content=f"Think step by step: {state['question']}")])
    return {"reasoning": r.content}

def act(state: AgentState):
    r = llm.invoke([HumanMessage(content=f"Answer clearly:\n{state['reasoning']}")])
    return {"answer": r.content}

graph = StateGraph(AgentState)
graph.add_node("reason", reason)
graph.add_node("act", act)
graph.set_entry_point("reason")
graph.add_edge("reason", "act")
graph.add_edge("act", END)

agent = graph.compile()

if __name__ == "__main__":
    print(agent.invoke({"question": "What is LangGraph?"})["answer"])
