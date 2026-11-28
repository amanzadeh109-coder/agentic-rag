"""A scripted stand-in for the LLM, used by tests (LLM_FAKE=true). No network.

Turn 1: calls search_reports with the user's question.
Turn 2: answers by citing the first source the tool returned, or refuses.
"""
import json
import re

from app.llm import LLMReply, Usage


def fake_stream(messages, tools):
    usage = Usage(prompt_tokens=100, completion_tokens=20, cost_usd=0.0)
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    question = next(m["content"] for m in reversed(messages) if m["role"] == "user")
    if False:
        yield  # makes this a generator even on the tool-call path
    if tools and not tool_msgs:
        args = json.dumps({"query": question})
        return LLMReply(tool_calls=[{"id": "call_1", "name": "search_reports", "arguments": args}],
                        usage=usage, model="fake")
    evidence = tool_msgs[-1]["content"] if tool_msgs else messages[-1]["content"]
    found = re.search(r"\[([\w.-]+\.pdf) p\.(\d+)\]", evidence)
    text = (f"The reports cover this [{found.group(1)} p.{found.group(2)}]." if found
            else "I could not find this in the reports.")
    for word in text.split(" "):
        yield word + " "
    return LLMReply(text=text, usage=usage, model="fake")