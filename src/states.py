from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field
from langchain_core.documents import Document
import operator


class GradeQuestion(BaseModel):
    score: str = Field(
        description="Question is about the specified topics? If yes -> 'Yes' if not -> 'No'"
    )

def overwrite(_, new):
    """Reducer: always replace the old value with the new one."""
    return new

def append_base_messages(old: List[BaseMessage], new: List[BaseMessage]):
    """Reducer: concatena listas de BaseMessage."""
    if old is None:
        old = []
    if new is None:
        new = []
    return old + new

# Estado de entrada del agente
class AgentInput(TypedDict):
    question: HumanMessage
    

class PrivateState(TypedDict):
    rephrased_question: Annotated[str, overwrite]
    reformulated_once: Annotated[int, overwrite]
    more_info_question: Annotated[str, overwrite]  # <-- debe ser str, no int
    proceed_to_generate: Annotated[bool, overwrite]
    news: Annotated[list[Document], operator.add]
    on_topic: Annotated[str, overwrite]
    rephrase_count: Annotated[int, overwrite]
    news_cycle_done: Annotated[bool, overwrite]  # <-- nuevo flag


# Estado de salida del agente
class AgentOutput(TypedDict):
    messages: Annotated[List[BaseMessage], append_base_messages]
    

class OverallState(AgentInput, PrivateState, AgentOutput):
    pass