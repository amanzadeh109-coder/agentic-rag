"""Streamlit client. Talks to the API only over HTTP, like any other client."""
import json
import os
import time

import httpx
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Annual Report Agent", page_icon="📊", layout="wide")

state = st.session_state
state.setdefault("user", "memo")
state.setdefault("conv_id", None)
state.setdefault("messages", [])


def call(method: str, path: str, **kw) -> httpx.Response | None:
    """Every request goes through here, so 'API is down' is handled in one place."""
    try:
        return httpx.request(method, API + path, headers={"X-User-Id": state.user},
                             timeout=kw.pop("timeout", 30), **kw)
    except httpx.HTTPError:
        st.error(f"The API at {API} is not reachable. Is `docker compose up` running?")
        return None


def show_sources(sources: list[dict]) -> None:
    for s in sources or []:
        label = f"📄 {s['file']} — page {s['page']}"
        if s.get("section"):
            label += f" · {s['section'][:60]}"
        if not s.get("verified", True):
            label += " ⚠️ not among the passages the agent read"
        with st.expander(label):
            st.markdown(s.get("text") or "_Passage not available._")


def show_steps(steps: list[dict]) -> None:
    if not steps:
        return
    searches = [s for s in steps if s["tool"] == "search_reports"]
    with st.expander(f"🔎 Agent: {len(steps)} tool call(s), {len(searches)} search(es)"):
        for s in steps:
            args = ", ".join(f"{k}={v}" for k, v in (s.get("args") or {}).items())
            mark = "✅" if s.get("ok") else "❌"
            st.markdown(f"{mark} **step {s['step']}** `{s['tool']}`({args}) → {s['results']} result(s)")


def load_conversation(conv_id: str) -> None:
    r = call("GET", f"/conversations/{conv_id}")
    if r is not None and r.status_code == 200:
        state.conv_id, state.messages = conv_id, r.json()["messages"]


# ---------------- sidebar: user, mode, budget, conversations, documents ----------------
with st.sidebar:
    st.title("📊 Annual Report Agent")
    state.user = st.text_input("User id (X-User-Id)", state.user)
    mode = st.radio("Answer mode", ["agent", "pipeline"], horizontal=True,
                    help="agent: the model searches as often as it needs. "
                         "pipeline: one search, one LLM call.")

    usage = call("GET", "/usage")
    if usage is not None and usage.status_code == 200:
        u = usage.json()
        st.progress(min(1.0, u["spent_usd"] / u["budget_usd"]),
                    text=f"Spent ${u['spent_usd']:.4f} of ${u['budget_usd']:.2f}")

    st.subheader("Conversations")
    if st.button("➕ New chat", use_container_width=True):
        state.conv_id, state.messages = None, []
    convs = call("GET", "/conversations")
    for c in (convs.json() if convs is not None and convs.status_code == 200 else [])[:15]:
        if st.button(c["title"], key=c["id"], use_container_width=True,
                     type="primary" if c["id"] == state.conv_id else "secondary"):
            load_conversation(c["id"])

    st.subheader("Documents")
    files = st.file_uploader("Upload PDF reports", type=["pdf"], accept_multiple_files=True)
    if files and st.button("Upload", use_container_width=True):
        ids = []
        for f in files:
            r = call("POST", "/documents", files={"file": (f.name, f.getvalue(), "application/pdf")},
                     timeout=120)
            if r is not None and r.status_code == 202:
                ids.append(r.json()["id"])
            elif r is not None:
                st.error(f"{f.name}: {r.json().get('detail', r.text)}")
        bar = st.progress(0.0, text="Processing...")
        while ids:  # poll the status until every upload is ready or failed
            done = 0
            for i in ids:
                d = call("GET", f"/documents/{i}")
                if d is not None and d.json()["status"] in ("ready", "failed"):
                    done += 1
            bar.progress(done / len(ids), text=f"Processed {done} of {len(ids)}")
            if done == len(ids):
                break
            time.sleep(2)
    docs = call("GET", "/documents")
    for d in (docs.json() if docs is not None and docs.status_code == 200 else []):
        icon = {"ready": "🟢", "failed": "🔴"}.get(d["status"], "🟡")
        col1, col2 = st.columns([5, 1])
        col1.caption(f"{icon} {d['filename']} · {d['status']}"
                     + (f" · {d['num_pages']} p." if d["num_pages"] else "")
                     + (f"\n\n{d['error']}" if d["error"] else ""))
        if col2.button("🗑", key="del" + d["id"], help="Delete everywhere"):
            call("DELETE", f"/documents/{d['id']}")
            st.rerun()

# ---------------- main: the conversation ----------------
for m in state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            show_steps(m.get("steps"))
            show_sources(m.get("sources"))
            if m.get("tokens"):
                st.caption(f"{m['tokens']} tokens · ${m['cost_usd']:.5f} · {m.get('model')}")

question = st.chat_input("Ask about the annual reports...")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        status = st.status("Thinking...", expanded=False)
        answer_box = st.empty()
        text, final = "", None
        body = {"question": question, "conversation_id": state.conv_id, "mode": mode, "stream": True}
        try:
            with httpx.stream("POST", API + "/chat", json=body, timeout=300,
                              headers={"X-User-Id": state.user}) as r:
                if r.status_code == 429:
                    status.update(label="Budget used up", state="error")
                    st.warning("💸 " + json.loads(r.read())["detail"])
                    st.stop()
                if r.status_code != 200:
                    status.update(label="Error", state="error")
                    st.error(f"The API answered {r.status_code}: {r.read().decode()[:300]}")
                    st.stop()
                event = None
                for line in r.iter_lines():  # SSE: "event: x" then "data: {...}"
                    if line.startswith("event:"):
                        event = line[6:].strip()
                    elif line.startswith("data:"):
                        data = json.loads(line[5:].strip())
                        if event == "step":
                            args = (data.get("args") or {}).get("query", "")
                            status.write(f"`{data['tool']}` {args} → {data['results']} result(s)")
                            status.update(label=f"Step {data['step']}: {data['tool']}")
                        elif event == "token":
                            text += data["text"]
                            answer_box.markdown(text + "▌")
                        elif event == "done":
                            final = data
                        elif event == "error":
                            status.update(label="Error", state="error")
                            st.error("⚠️ " + data["message"])
        except httpx.HTTPError:
            status.update(label="API down", state="error")
            st.error(f"Lost the connection to the API at {API}.")
            st.stop()

        if final:
            status.update(label=f"Done in {len(final['steps'])} step(s), {final['seconds']}s",
                          state="complete")
            answer_box.markdown(final["answer"])
            if final["refused"]:
                st.info("ℹ️ The reports do not seem to contain this. Try rephrasing, "
                        "or upload the report that would answer it.")
            show_steps(final["steps"])
            show_sources(final["sources"])
            tokens = final["usage"]["prompt_tokens"] + final["usage"]["completion_tokens"]
            st.caption(f"{tokens} tokens · ${final['usage']['usd']:.5f} · {final['model']}")
            state.conv_id = final["conversation_id"]
            state.messages.append({"role": "assistant", "content": final["answer"],
                                   "sources": final["sources"], "steps": final["steps"],
                                   "tokens": tokens, "cost_usd": final["usage"]["usd"],
                                   "model": final["model"]})