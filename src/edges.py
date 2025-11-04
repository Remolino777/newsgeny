from typing import Literal
from src.states import PrivateState 



def on_topic_router(state: PrivateState) -> Literal["yes", "no"]:
    print("Entering on_topic_router")
    on_topic = state.get("on_topic", "").strip().lower()
    if on_topic == "yes":
        print("Routing to yes branch")
        return "yes"
    else:
        print("Routing to no branch")
        return "no"

def news_router(state: PrivateState) -> str:
    if state.get("reformulated_once") == 1 and state.get("news_cycle_done", False):
        return "on_topic_response"
    elif state.get("reformulated_once") == 1:
        return "get_more_news"
    else:
        return "on_topic_response"


