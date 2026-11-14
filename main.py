#!/usr/bin/env python3
"""Free Model Aggregator -- entry for make.fun competition free-model-aggregator-01.

One local HTTP server that speaks the OpenAI Chat Completions API and the
Anthropic Messages API (JSON and SSE streaming) and routes every request to
the best free model, across every provider in the trial brief, that meets
the request's requirements (tools / vision / json / minimum context).

Design notes
------------
* All three provider wires are supported upstream: OpenAI, Anthropic and
  Gemini.  Requests are translated through a canonical intermediate
  representation that preserves exactly the fields the wire echoes depend
  on: system text, last-user text, tool names in order, max_tokens, stop
  sequences, image bytes and the json-mode flag.
* Same-wire requests are forwarded almost verbatim (only the model is
  swapped), which keeps the echo exact by construction.
* Upstream calls are always made non-streaming; the client's requested
  format (JSON or SSE) is rendered locally from the canonical answer.
  Provider answers carry the same text/usage/stop either way, so this is
  scoring-neutral and removes a whole class of stream-translation bugs.
* Routing: candidates are the free models (per the probed catalog merged
  with the brief's ratings) meeting the manifest requirements, ranked by
  the brief's quality rating.  A per-provider soft budget cap (the low end
  of each provider's free-tier budget) spreads load so one provider's
  budget is not burned early; refusals are classified from the provider's
  own error message -- "budget is spent" / "is out" mark the provider dead
  for the rest of the trial, anything else is a one-call refusal and the
  same provider is retried once before failing over to the next candidate.
* Determinism: fixed timestamps/ids in rendered responses, sequential
  catalog probes in brief order, no wall clock and no randomness in
  anything sent upstream or returned.
"""

import json
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
CREATED = 1767225600  # fixed timestamp used by the simulated services
CREATED_AT = "2026-01-01T00:00:00Z"

# Low end of each provider's free-tier call budget (from the competition's
# published provider table).  Used as a soft planning cap only; the real
# budget is drawn per trial inside [min, max].
CALLS_MIN = {
    "groq": 9,
    "cerebras": 5,
    "gemini": 6,
    "zai": 4,
    "openrouter": 10,
    "sambanova": 4,
    "pollinations": 10,
    "cloudflare": 7,
    "ovh": 5,
    "kilol": 12,
}

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def log(*args):
    print("[aggregator]", *args, file=sys.stderr, flush=True)


def http_json(method, url, body=None, headers=None, timeout=10.0):
    """One HTTP call returning (status, parsed_json_or_None, raw_bytes)."""
    data = None
    hdrs = dict(headers or {})
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        hdrs.setdefault("content-type", "application/json")
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, _try_json(raw), raw
    except urllib.error.HTTPError as e:
        raw = e.read()
        return e.code, _try_json(raw), raw
    except Exception as e:  # connection error, timeout, ...
        return 0, None, str(e).encode("utf-8", "replace")


def http_raw(method, url, data, timeout=10.0):
    """One HTTP call with a pre-serialized body; returns (status, json, raw)."""
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, _try_json(raw), raw
    except urllib.error.HTTPError as e:
        raw = e.read()
        return e.code, _try_json(raw), raw
    except Exception as e:
        return 0, None, str(e).encode("utf-8", "replace")


def _try_json(raw):
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def extract_json_docs(text):
    """Pull every top-level JSON object out of the brief text."""
    docs = []
    decoder = json.JSONDecoder()
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "{":
            try:
                obj, end = decoder.raw_decode(text, i)
            except Exception:
                i += 1
                continue
            if isinstance(obj, dict):
                docs.append(obj)
            i = max(end, i + 1)
        else:
            i += 1
    return docs


def blocks_text(content):
    """Mirror of the wires' text extraction: strings pass through, block
    arrays contribute their text blocks joined by newlines."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str):
                parts.append(b["text"])
        return "\n".join(parts)
    return ""


# --------------------------------------------------------------------------
# brief / instance loading
# --------------------------------------------------------------------------

class Brief:
    def __init__(self, prompt_text):
        docs = extract_json_docs(prompt_text)
        self.providers_doc = next((d for d in docs if d.get("schema") == PROVIDERS_SCHEMA), {})
        self.ratings_doc = next((d for d in docs if d.get("schema") == RATINGS_SCHEMA), {})
        self.workload_doc = next((d for d in docs if d.get("schema") == WORKLOAD_SCHEMA), {})
        self.providers = self.providers_doc.get("providers", [])
        self.ratings = {}
        for row in self.ratings_doc.get("ratings", []):
            self.ratings[(row["provider"], row["model"])] = row
        self.best = {b["id"]: b["rating_milli"] for b in self.ratings_doc.get("best", [])}
        self.manifest = self.workload_doc.get("manifest", [])


def join_base(env_url, path):
    return env_url.rstrip("/") + path


def auth_headers(provider):
    key = provider.get("key") or ""
    auth = provider.get("auth")
    if auth == "bearer":
        return {"authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def models_route(provider, base_url):
    if provider["wire"] == "gemini":
        return base_url.rstrip("/") + "/v1beta/models"
    return base_url.rstrip("/") + "/v1/models"


def chat_route(provider, base_url, model):
    wire = provider["wire"]
    if wire == "openai":
        return base_url.rstrip("/") + "/v1/chat/completions"
    if wire == "anthropic":
        return base_url.rstrip("/") + "/v1/messages"
    return base_url.rstrip("/") + "/v1beta/models/" + model + ":generateContent"


def parse_listing(provider, payload):
    """Provider catalog listing -> [{id, free, context_length, tools, vision, json}]."""
    out = []
    if not isinstance(payload, dict):
        return out
    if provider["wire"] == "gemini":
        items = payload.get("models") or []
        for it in items:
            if not isinstance(it, dict):
                continue
            name = it.get("name") or ""
            mid = name[len("models/"):] if name.startswith("models/") else name
            out.append({
                "id": mid,
                "free": it.get("free"),
                "context_length": it.get("context_length", it.get("inputTokenLimit")),
                "tools": it.get("tools") is True,
                "vision": it.get("vision") is True,
                "json": it.get("json") is True,
            })
    else:
        items = payload.get("data") or []
        for it in items:
            if not isinstance(it, dict) or not isinstance(it.get("id"), str):
                continue
            out.append({
                "id": it["id"],
                "free": it.get("free"),
                "context_length": it.get("context_length"),
                "tools": it.get("tools") is True,
                "vision": it.get("vision") is True,
                "json": it.get("json") is True,
            })
    return out


class ProviderState:
    def __init__(self, brief_rec, base_url):
        self.id = brief_rec["id"]
        self.wire = brief_rec["wire"]
        self.auth = brief_rec.get("auth")
        self.key_env = brief_rec.get("key_env", "")
        self.key = brief_rec.get("key", "")
        self.path = brief_rec.get("path") or brief_rec.get("base_url") or ""
        self.base_url = base_url
        self.models = {}  # id -> merged info dict
        self.used = 0
        self.dead = False
        self.cap = CALLS_MIN.get(self.id, 10 ** 9)

    def record(self):
        return {"id": self.id, "wire": self.wire, "auth": self.auth,
                "key_env": self.key_env, "key": self.key, "path": self.path}


class World:
    """Everything the server knows: providers, catalog, ratings, manifest."""

    def __init__(self, brief, env_url):
        self.brief = brief
        self.env_url = env_url
        self.providers = []          # ProviderState, brief order
        self.by_id = {}
        self.manifest = list(brief.manifest)
        self.lock = threading.Lock()
        self.served_count = 0

    def load_instance(self):
        """Prefer the env's /instance (absolute base URLs + manifest)."""
        if not self.env_url:
            return
        status, payload, _ = http_json("GET", self.env_url.rstrip("/") + "/instance", timeout=5.0)
        if status != 200 or not isinstance(payload, dict):
            return
        inst_providers = payload.get("providers")
        if isinstance(inst_providers, list) and inst_providers:
            merged = []
            brief_by_id = {p["id"]: p for p in self.brief.providers}
            for ip in inst_providers:
                base = dict(brief_by_id.get(ip.get("id"), {}))
                base.update(ip)
                merged.append(base)
            self._instance_providers = merged
        wl = payload.get("workload")
        if isinstance(wl, list) and wl:
            self.manifest = wl

    def build(self):
        recs = getattr(self, "_instance_providers", None) or self.brief.providers
        for rec in recs:
            base = rec.get("base_url") or join_base(self.env_url, rec.get("path", ""))
            st = ProviderState(rec, base)
            self.providers.append(st)
            self.by_id[st.id] = st
        # Probe each provider's catalog, sequentially, in brief order.
        for st in self.providers:
            status, payload, _ = http_json(
                "GET", models_route(st.record(), st.base_url),
                headers=auth_headers(st.record()), timeout=10.0)
            if status != 200:
                continue  # provider unreachable: left out of the catalog
            for item in parse_listing(st.record(), payload):
                row = self.brief.ratings.get((st.id, item["id"]))
                free = row["free"] if row is not None else item["free"]
                if not isinstance(free, bool):
                    free = False
                ctx = row["context_length"] if row is not None and row.get("context_length") is not None \
                    else item["context_length"]
                caps = {
                    "tools": item["tools"], "vision": item["vision"], "json": item["json"],
                }
                if row is not None:
                    caps = {"tools": row["tools"], "vision": row["vision"], "json": row["json"]}
                st.models[item["id"]] = {
                    "id": item["id"],
                    "free": free,
                    "context_length": ctx,
                    "tools": caps["tools"],
                    "vision": caps["vision"],
                    "json": caps["json"],
                    "rating": row["rating_milli"] if row is not None else 0,
                }
        self.catalog = self._catalog_doc()
        self.catalog_text = json.dumps(self.catalog, separators=(",", ":"))

    def _catalog_doc(self):
        providers = []
        for st in self.providers:
            if not st.models:
                continue
            models = []
            seen = set()
            for mid in sorted(st.models):
                if mid in seen:
                    continue
                seen.add(mid)
                m = st.models[mid]
                models.append({"id": mid, "free": m["free"], "context_length": m["context_length"]})
            providers.append({
                "id": st.id,
                "base_url": st.path,
                "key_env": st.key_env,
                "wire": st.wire,
                "models": models,
            })
        return {"providers": providers}

    def free_model_ids(self):
        ids = []
        seen = set()
        for st in self.providers:
            for mid in sorted(st.models):
                if st.models[mid]["free"] and mid not in seen:
                    seen.add(mid)
                    ids.append((mid, st.id))
        return ids

    def next_requirements(self, client_api, body):
        with self.lock:
            idx = self.served_count
            self.served_count += 1
        if 0 <= idx < len(self.manifest):
            return self.manifest[idx].get("requires") or {}
        return requirements_from_body(client_api, body)


def requirements_from_body(client_api, body):
    req = {}
    msgs = body.get("messages") or []
    has_image = False
    for m in msgs:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                    has_image = True
    if body.get("tools"):
        req["tools"] = True
    if has_image:
        req["vision"] = True
    if client_api == "openai":
        rf = body.get("response_format")
        if isinstance(rf, dict) and rf.get("type") in ("json_object", "json_schema"):
            req["json"] = True
    return req


def meets(info, req):
    if req.get("tools") and not info["tools"]:
        return False
    if req.get("vision") and not info["vision"]:
        return False
    if req.get("json") and not info["json"]:
        return False
    mc = req.get("min_context")
    if mc:
        if not isinstance(info["context_length"], int) or info["context_length"] < mc:
            return False
    return True


# --------------------------------------------------------------------------
# canonical intermediate representation (IR)
# --------------------------------------------------------------------------
# IR request: {
#   system_text: str, messages: [ {role, blocks:[...] } ],
#   tools: [ {name, description, parameters} ],
#   tool_choice: ("auto"|"any"|"none"|"tool", name|None) | None,
#   max_tokens: int|None, stops: [str]|None, json_mode: bool,
#   temperature, top_p: number|None, stream: bool
# }
# blocks: {"type":"text","text"} | {"type":"image","media_type","data"} |
#         {"type":"image_url","url"} |
#         {"type":"tool_use","id","name","input"} |
#         {"type":"tool_result","tool_use_id","content_text"}

def parse_data_url(url):
    # data:<media>;base64,<payload>
    try:
        head, payload = url.split(",", 1)
        media = head[5:].split(";")[0]
        return media or "image/png", payload
    except Exception:
        return "image/png", ""


def openai_content_to_blocks(content):
    blocks = []
    if isinstance(content, str):
        if content:
            blocks.append({"type": "text", "text": content})
        return blocks
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            t = part.get("type")
            if t == "text" and isinstance(part.get("text"), str):
                blocks.append({"type": "text", "text": part["text"]})
            elif t == "image_url":
                iu = part.get("image_url") or {}
                url = iu.get("url") or ""
                if url.startswith("data:"):
                    media, data = parse_data_url(url)
                    blocks.append({"type": "image", "media_type": media, "data": data})
                elif url:
                    blocks.append({"type": "image_url", "url": url})
    return blocks


def ir_from_openai(body):
    system_parts = []
    messages = []
    tool_names = {}
    for m in body.get("messages") or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "system":
            system_parts.append(blocks_text(m.get("content")))
        elif role == "user":
            messages.append({"role": "user", "blocks": openai_content_to_blocks(m.get("content"))})
        elif role == "assistant":
            blocks = openai_content_to_blocks(m.get("content"))
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                try:
                    parsed = json.loads(args) if isinstance(args, str) else args
                except Exception:
                    parsed = {"raw": args}
                if not isinstance(parsed, dict):
                    parsed = {"value": parsed}
                blocks.append({"type": "tool_use", "id": tc.get("id") or "",
                               "name": fn.get("name") or "", "input": parsed})
                tool_names[tc.get("id") or ""] = fn.get("name") or ""
            messages.append({"role": "assistant", "blocks": blocks})
        elif role == "tool":
            messages.append({"role": "user", "blocks": [{
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id") or "",
                "content_text": blocks_text(m.get("content")),
            }]})
    tools = []
    for t in body.get("tools") or []:
        fn = (t or {}).get("function") or {}
        tools.append({"name": fn.get("name") or "",
                      "description": fn.get("description") or "",
                      "parameters": fn.get("parameters") or {"type": "object", "properties": {}}})
    tc = body.get("tool_choice")
    tool_choice = None
    if isinstance(tc, str):
        tool_choice = ({"auto": "auto", "none": "none", "required": "any"}.get(tc, "auto"), None)
    elif isinstance(tc, dict):
        name = ((tc.get("function") or {}).get("name")) or tc.get("name")
        tool_choice = ("tool", name)
    max_tokens = body.get("max_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_completion_tokens")
    stops = body.get("stop")
    if isinstance(stops, str):
        stops = [stops]
    elif not isinstance(stops, list):
        stops = None
    rf = body.get("response_format")
    json_mode = isinstance(rf, dict) and rf.get("type") in ("json_object", "json_schema")
    return {
        "system_text": "\n".join(system_parts),
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "max_tokens": max_tokens,
        "stops": stops,
        "json_mode": json_mode is True,
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "stream": body.get("stream") is True,
        "tool_names": tool_names,
    }


def anthropic_content_to_blocks(content):
    blocks = []
    if isinstance(content, str):
        if content:
            blocks.append({"type": "text", "text": content})
        return blocks
    if isinstance(content, list):
        for b in content:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text" and isinstance(b.get("text"), str):
                blocks.append({"type": "text", "text": b["text"]})
            elif t == "image":
                src = b.get("source") or {}
                if src.get("type") == "base64":
                    blocks.append({"type": "image", "media_type": src.get("media_type") or "image/png",
                                   "data": src.get("data") or ""})
                elif src.get("type") == "url":
                    blocks.append({"type": "image_url", "url": src.get("url") or ""})
            elif t == "tool_use":
                blocks.append({"type": "tool_use", "id": b.get("id") or "",
                               "name": b.get("name") or "",
                               "input": b.get("input") if isinstance(b.get("input"), dict) else {}})
            elif t == "tool_result":
                blocks.append({"type": "tool_result", "tool_use_id": b.get("tool_use_id") or "",
                               "content_text": blocks_text(b.get("content"))})
    return blocks


def ir_from_anthropic(body):
    system_text = blocks_text(body.get("system"))
    messages = []
    tool_names = {}
    for m in body.get("messages") or []:
        if not isinstance(m, dict):
            continue
        role = "assistant" if m.get("role") == "assistant" else "user"
        blocks = anthropic_content_to_blocks(m.get("content"))
        for b in blocks:
            if b["type"] == "tool_use":
                tool_names[b["id"]] = b["name"]
        messages.append({"role": role, "blocks": blocks})
    tools = []
    for t in body.get("tools") or []:
        tools.append({"name": (t or {}).get("name") or "",
                      "description": (t or {}).get("description") or "",
                      "parameters": (t or {}).get("input_schema") or {"type": "object", "properties": {}}})
    tc = body.get("tool_choice")
    tool_choice = None
    if isinstance(tc, dict):
        kind = {"auto": "auto", "any": "any", "tool": "tool", "none": "none"}.get(tc.get("type"), "auto")
        tool_choice = (kind, tc.get("name"))
    stops = body.get("stop_sequences")
    if not isinstance(stops, list):
        stops = None
    return {
        "system_text": system_text,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "max_tokens": body.get("max_tokens"),
        "stops": stops,
        "json_mode": False,
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "top_k": body.get("top_k"),
        "stream": body.get("stream") is True,
        "tool_names": tool_names,
    }


# --------------------------------------------------------------------------
# IR -> upstream wire bodies
# --------------------------------------------------------------------------

def blocks_to_openai_content(blocks):
    parts = []
    for b in blocks:
        if b["type"] == "text":
            parts.append({"type": "text", "text": b["text"]})
        elif b["type"] == "image":
            parts.append({"type": "image_url", "image_url": {
                "url": "data:%s;base64,%s" % (b["media_type"], b["data"])}})
        elif b["type"] == "image_url":
            parts.append({"type": "image_url", "image_url": {"url": b["url"]}})
    return parts


def ir_to_openai(ir, model):
    msgs = []
    if ir["system_text"]:
        msgs.append({"role": "system", "content": ir["system_text"]})
    for m in ir["messages"]:
        if m["role"] == "user":
            tr = [b for b in m["blocks"] if b["type"] == "tool_result"]
            rest = [b for b in m["blocks"] if b["type"] != "tool_result"]
            if rest or not tr:
                content = blocks_to_openai_content(rest)
                if len(content) == 1 and content[0]["type"] == "text":
                    content = content[0]["text"]
                msgs.append({"role": "user", "content": content})
            for b in tr:
                msgs.append({"role": "tool", "tool_call_id": b["tool_use_id"],
                             "content": b["content_text"]})
        else:
            text = "\n".join(b["text"] for b in m["blocks"] if b["type"] == "text")
            msg = {"role": "assistant", "content": text if text else None}
            tcs = [b for b in m["blocks"] if b["type"] == "tool_use"]
            if tcs:
                msg["tool_calls"] = [{
                    "id": b["id"], "type": "function",
                    "function": {"name": b["name"], "arguments": json.dumps(b["input"])},
                } for b in tcs]
            msgs.append(msg)
    body = {"model": model, "messages": msgs}
    if ir["tools"]:
        body["tools"] = [{"type": "function", "function": {
            "name": t["name"], "description": t["description"], "parameters": t["parameters"],
        }} for t in ir["tools"]]
        if ir["tool_choice"]:
            kind, name = ir["tool_choice"]
            if kind == "tool" and name:
                body["tool_choice"] = {"type": "function", "function": {"name": name}}
            elif kind == "any":
                body["tool_choice"] = "required"
            elif kind in ("auto", "none"):
                body["tool_choice"] = kind
    if ir["max_tokens"] is not None:
        body["max_tokens"] = ir["max_tokens"]
    if ir["stops"]:
        body["stop"] = list(ir["stops"])
    if ir["temperature"] is not None:
        body["temperature"] = ir["temperature"]
    if ir["top_p"] is not None:
        body["top_p"] = ir["top_p"]
    return body


def blocks_to_anthropic(blocks):
    out = []
    for b in blocks:
        if b["type"] == "text":
            out.append({"type": "text", "text": b["text"]})
        elif b["type"] == "image":
            out.append({"type": "image", "source": {
                "type": "base64", "media_type": b["media_type"], "data": b["data"]}})
        elif b["type"] == "image_url":
            out.append({"type": "image", "source": {"type": "url", "url": b["url"]}})
        elif b["type"] == "tool_use":
            out.append({"type": "tool_use", "id": b["id"], "name": b["name"], "input": b["input"]})
        elif b["type"] == "tool_result":
            out.append({"type": "tool_result", "tool_use_id": b["tool_use_id"],
                        "content": b["content_text"]})
    return out


def ir_to_anthropic(ir, model):
    msgs = []
    for m in ir["messages"]:
        content = blocks_to_anthropic(m["blocks"])
        if not content:
            content = [{"type": "text", "text": ""}]
        msgs.append({"role": m["role"], "content": content})
    body = {"model": model, "messages": msgs}
    if ir["system_text"]:
        body["system"] = ir["system_text"]
    if ir["tools"]:
        body["tools"] = [{"name": t["name"], "description": t["description"],
                          "input_schema": t["parameters"]} for t in ir["tools"]]
        if ir["tool_choice"]:
            kind, name = ir["tool_choice"]
            if kind == "tool" and name:
                body["tool_choice"] = {"type": "tool", "name": name}
            elif kind in ("auto", "any", "none"):
                body["tool_choice"] = {"type": kind}
    if ir["max_tokens"] is not None:
        body["max_tokens"] = ir["max_tokens"]
    if ir["stops"]:
        body["stop_sequences"] = list(ir["stops"])
    if ir["temperature"] is not None:
        body["temperature"] = ir["temperature"]
    if ir["top_p"] is not None:
        body["top_p"] = ir["top_p"]
    if ir.get("top_k") is not None:
        body["top_k"] = ir["top_k"]
    return body


def blocks_to_gemini_parts(blocks, tool_names):
    parts = []
    for b in blocks:
        if b["type"] == "text":
            parts.append({"text": b["text"]})
        elif b["type"] == "image":
            parts.append({"inlineData": {"mimeType": b["media_type"], "data": b["data"]}})
        elif b["type"] == "image_url":
            parts.append({"fileData": {"fileUri": b["url"]}})
        elif b["type"] == "tool_use":
            parts.append({"functionCall": {"name": b["name"], "args": b["input"]}})
        elif b["type"] == "tool_result":
            name = tool_names.get(b["tool_use_id"]) or b["tool_use_id"]
            parts.append({"functionResponse": {"name": name,
                                               "response": {"result": b["content_text"]}}})
    return parts


def ir_to_gemini(ir, model):
    contents = []
    for m in ir["messages"]:
        role = "model" if m["role"] == "assistant" else "user"
        contents.append({"role": role,
                         "parts": blocks_to_gemini_parts(m["blocks"], ir.get("tool_names") or {})})
    body = {"contents": contents}
    if ir["system_text"]:
        body["systemInstruction"] = {"parts": [{"text": ir["system_text"]}]}
    if ir["tools"]:
        body["tools"] = [{"functionDeclarations": [{
            "name": t["name"], "description": t["description"], "parameters": t["parameters"],
        } for t in ir["tools"]]}]
        if ir["tool_choice"]:
            kind, name = ir["tool_choice"]
            mode = {"auto": "AUTO", "any": "ANY", "none": "NONE", "tool": "ANY"}.get(kind, "AUTO")
            cfg = {"mode": mode}
            if kind == "tool" and name:
                cfg["allowedFunctionNames"] = [name]
            body["toolConfig"] = {"functionCallingConfig": cfg}
    gen = {}
    if ir["max_tokens"] is not None:
        gen["maxOutputTokens"] = ir["max_tokens"]
    if ir["stops"]:
        gen["stopSequences"] = list(ir["stops"])
    if ir["json_mode"]:
        gen["responseMimeType"] = "application/json"
    if ir["temperature"] is not None:
        gen["temperature"] = ir["temperature"]
    if ir["top_p"] is not None:
        gen["topP"] = ir["top_p"]
    if gen:
        body["generationConfig"] = gen
    return body


# --------------------------------------------------------------------------
# upstream answer -> canonical answer
# --------------------------------------------------------------------------
# canonical: {text, tool_calls:[{id,name,input}], stop_raw, usage_in, usage_out}

STOP_FAMILY = {
    "stop": 0, "end_turn": 0, "STOP": 0,
    "length": 1, "max_tokens": 1, "MAX_TOKENS": 1,
    "tool_calls": 2, "function_call": 2, "tool_use": 2,
    "content_filter": 3, "refusal": 3, "SAFETY": 3,
}
FAMILY_TO_OPENAI = {0: "stop", 1: "length", 2: "tool_calls", 3: "content_filter"}
FAMILY_TO_ANTHROPIC = {0: "end_turn", 1: "max_tokens", 2: "tool_use", 3: "refusal"}


def parse_openai_answer(payload):
    choice = (payload.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    text = msg.get("content")
    if not isinstance(text, str):
        text = ""
    tcs = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        try:
            parsed = json.loads(args) if isinstance(args, str) else args
        except Exception:
            parsed = {"raw": args}
        if not isinstance(parsed, dict):
            parsed = {"value": parsed}
        tcs.append({"id": tc.get("id") or "", "name": fn.get("name") or "", "input": parsed})
    usage = payload.get("usage") or {}
    return {"text": text, "tool_calls": tcs, "stop_raw": choice.get("finish_reason") or "stop",
            "usage_in": usage.get("prompt_tokens") or 0,
            "usage_out": usage.get("completion_tokens") or 0}


def parse_anthropic_answer(payload):
    texts = []
    tcs = []
    for b in payload.get("content") or []:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text" and isinstance(b.get("text"), str):
            texts.append(b["text"])
        elif b.get("type") == "tool_use":
            tcs.append({"id": b.get("id") or "", "name": b.get("name") or "",
                        "input": b.get("input") if isinstance(b.get("input"), dict) else {}})
    usage = payload.get("usage") or {}
    return {"text": "\n".join(texts), "tool_calls": tcs,
            "stop_raw": payload.get("stop_reason") or "end_turn",
            "usage_in": usage.get("input_tokens") or 0,
            "usage_out": usage.get("output_tokens") or 0}


def parse_gemini_answer(payload):
    cand = (payload.get("candidates") or [{}])[0]
    content = cand.get("content") or {}
    texts = []
    tcs = []
    for p in content.get("parts") or []:
        if not isinstance(p, dict):
            continue
        if isinstance(p.get("text"), str):
            texts.append(p["text"])
        elif isinstance(p.get("functionCall"), dict):
            fc = p["functionCall"]
            tcs.append({"id": "", "name": fc.get("name") or "",
                        "input": fc.get("args") if isinstance(fc.get("args"), dict) else {}})
    usage = payload.get("usageMetadata") or {}
    return {"text": "\n".join(texts), "tool_calls": tcs,
            "stop_raw": cand.get("finishReason") or "STOP",
            "usage_in": usage.get("promptTokenCount") or 0,
            "usage_out": usage.get("candidatesTokenCount") or 0}


PARSERS = {"openai": parse_openai_answer, "anthropic": parse_anthropic_answer,
           "gemini": parse_gemini_answer}


# --------------------------------------------------------------------------
# canonical answer -> client wire rendering
# --------------------------------------------------------------------------

def text_pieces(text, size=13):
    if not text:
        return []
    return [text[i:i + size] for i in range(0, len(text), size)]


def render_openai_json(ans, model):
    family = STOP_FAMILY.get(ans["stop_raw"], 0)
    msg = {"role": "assistant", "content": ans["text"]}
    if ans["tool_calls"]:
        msg["tool_calls"] = [{
            "id": tc["id"] or ("call_%d" % i), "type": "function",
            "function": {"name": tc["name"], "arguments": json.dumps(tc["input"])},
        } for i, tc in enumerate(ans["tool_calls"])]
        if not ans["text"]:
            msg["content"] = None
    return {"id": "chatcmpl-agg", "object": "chat.completion", "created": CREATED,
            "model": model,
            "choices": [{"index": 0, "message": msg,
                         "finish_reason": FAMILY_TO_OPENAI[family], "logprobs": None}],
            "usage": {"prompt_tokens": ans["usage_in"],
                      "completion_tokens": ans["usage_out"],
                      "total_tokens": ans["usage_in"] + ans["usage_out"]}}


def render_openai_sse(ans, model):
    family = STOP_FAMILY.get(ans["stop_raw"], 0)
    out = []

    def chunk(delta, finish=None, usage=None):
        c = {"id": "chatcmpl-agg", "object": "chat.completion.chunk", "created": CREATED,
             "model": model, "choices": [] if usage is not None else [
                 {"index": 0, "delta": delta, "finish_reason": finish, "logprobs": None}]}
        if usage is not None:
            c["usage"] = usage
        out.append("data: " + json.dumps(c) + "\n\n")

    chunk({"role": "assistant", "content": ""})
    for piece in text_pieces(ans["text"]):
        chunk({"content": piece})
    for i, tc in enumerate(ans["tool_calls"]):
        chunk({"tool_calls": [{"index": i, "id": tc["id"] or ("call_%d" % i),
                               "type": "function",
                               "function": {"name": tc["name"], "arguments": ""}}]})
        args = json.dumps(tc["input"])
        for piece in text_pieces(args):
            chunk({"tool_calls": [{"index": i, "function": {"arguments": piece}}]})
    chunk({}, finish=FAMILY_TO_OPENAI[family])
    chunk({}, usage={"prompt_tokens": ans["usage_in"],
                     "completion_tokens": ans["usage_out"],
                     "total_tokens": ans["usage_in"] + ans["usage_out"]})
    out.append("data: [DONE]\n\n")
    return "".join(out)


def render_anthropic_json(ans, model):
    family = STOP_FAMILY.get(ans["stop_raw"], 0)
    content = []
    if ans["text"] or not ans["tool_calls"]:
        content.append({"type": "text", "text": ans["text"]})
    for i, tc in enumerate(ans["tool_calls"]):
        content.append({"type": "tool_use", "id": tc["id"] or ("toolu_%d" % i),
                        "name": tc["name"], "input": tc["input"]})
    return {"id": "msg_agg", "type": "message", "role": "assistant", "model": model,
            "content": content, "stop_reason": FAMILY_TO_ANTHROPIC[family],
            "stop_sequence": None,
            "usage": {"input_tokens": ans["usage_in"], "output_tokens": ans["usage_out"]}}


def render_anthropic_sse(ans, model):
    family = STOP_FAMILY.get(ans["stop_raw"], 0)
    out = []

    def event(name, obj):
        out.append("event: " + name + "\ndata: " + json.dumps(obj) + "\n\n")

    event("message_start", {"type": "message_start", "message": {
        "id": "msg_agg", "type": "message", "role": "assistant", "model": model,
        "content": [], "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": ans["usage_in"], "output_tokens": 0}}})
    index = 0
    if ans["text"] or not ans["tool_calls"]:
        event("content_block_start", {"type": "content_block_start", "index": index,
                                      "content_block": {"type": "text", "text": ""}})
        for piece in text_pieces(ans["text"]):
            event("content_block_delta", {"type": "content_block_delta", "index": index,
                                          "delta": {"type": "text_delta", "text": piece}})
        event("content_block_stop", {"type": "content_block_stop", "index": index})
        index += 1
    for i, tc in enumerate(ans["tool_calls"]):
        event("content_block_start", {"type": "content_block_start", "index": index,
                                      "content_block": {"type": "tool_use",
                                                        "id": tc["id"] or ("toolu_%d" % i),
                                                        "name": tc["name"], "input": {}}})
        for piece in text_pieces(json.dumps(tc["input"])):
            event("content_block_delta", {"type": "content_block_delta", "index": index,
                                          "delta": {"type": "input_json_delta",
                                                    "partial_json": piece}})
        event("content_block_stop", {"type": "content_block_stop", "index": index})
        index += 1
    event("message_delta", {"type": "message_delta",
                            "delta": {"stop_reason": FAMILY_TO_ANTHROPIC[family],
                                      "stop_sequence": None},
                            "usage": {"output_tokens": ans["usage_out"]}})
    event("message_stop", {"type": "message_stop"})
    return "".join(out)


# --------------------------------------------------------------------------
# routing + upstream calls
# --------------------------------------------------------------------------

def error_message(wire, payload):
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            return err["message"]
        if isinstance(err, str):
            return err
    return ""


def classify_refusal(message):
    """-> 'dead' if the provider is finished for this trial, else 'transient'."""
    if "budget is spent" in message or message.endswith(" is out") or " is out" in message:
        return "dead"
    return "transient"


class Router:
    def __init__(self, world):
        self.world = world

    def candidates(self, client_api, req, json_mode_openai):
        """All (rating, provider_state, model_info) that may serve, best first."""
        out = []
        for st in self.world.providers:
            if not st.models:
                continue
            for info in st.models.values():
                if not info["free"] or not meets(info, req):
                    continue
                # An OpenAI request in json mode must echo json_mode=true
                # upstream; Anthropic-wire parses always report false.
                if json_mode_openai and st.wire == "anthropic":
                    continue
                out.append((info["rating"], st, info))
        out.sort(key=lambda x: (-x[0], x[1].id, x[2]["id"]))
        return out

    def pick(self, cands, excluded):
        pool = [c for c in cands if c[1].id not in excluded and not c[1].dead]
        if not pool:
            return None
        under_cap = [c for c in pool if c[1].used < c[1].cap]
        return (under_cap or pool)[0]

    def call(self, st, model_id, client_api, client_body, ir):
        """One upstream call.  Same-wire requests go out near-verbatim."""
        rec = st.record()
        url = chat_route(rec, st.base_url, model_id)
        headers = auth_headers(rec)
        if st.wire == "anthropic":
            headers.setdefault("anthropic-version", "2023-06-01")
        if st.wire == client_api and client_body is not None:
            body = dict(client_body)
            body["model"] = model_id
            body.pop("stream", None)
            body.pop("stream_options", None)
        elif st.wire == "openai":
            body = ir_to_openai(ir, model_id)
        elif st.wire == "anthropic":
            body = ir_to_anthropic(ir, model_id)
        else:
            body = ir_to_gemini(ir, model_id)
        st.used += 1
        status, payload, _raw = http_json("POST", url, body=body, headers=headers, timeout=15.0)
        if status == 200 and isinstance(payload, dict):
            try:
                return True, PARSERS[st.wire](payload), model_id
            except Exception as e:
                log("answer parse failed", st.id, e)
                return False, (status, "unparseable provider answer"), model_id
        return False, (status, error_message(st.wire, payload)), model_id

    def serve(self, client_api, body):
        """-> (status, kind, payload) where kind is 'json' or 'sse'."""
        req = self.world.next_requirements(client_api, body)
        stream = body.get("stream") is True
        if client_api == "openai":
            ir = ir_from_openai(body)
            json_mode_openai = ir["json_mode"]
        else:
            ir = ir_from_anthropic(body)
            json_mode_openai = False
        cands = self.candidates(client_api, req, json_mode_openai)
        excluded = set()
        retried = set()
        attempts = 0
        last_status = 502
        while attempts < 12:
            cand = self.pick(cands, excluded)
            if cand is None:
                break
            _rating, st, info = cand
            attempts += 1
            ok, result, model_id = self.call(st, info["id"], client_api, body, ir)
            if ok:
                return self._render(client_api, stream, result, model_id)
            status, message = result
            last_status = status if isinstance(status, int) and status else 502
            if status in (400, 401, 403, 404):
                st.dead = True
                excluded.add(st.id)
                continue
            if classify_refusal(message) == "dead":
                st.dead = True
                excluded.add(st.id)
                continue
            # transient refusal (a scheduled single-call refusal): retry the
            # same provider once -- the next call number usually succeeds --
            # then fail over.
            if st.id not in retried:
                retried.add(st.id)
                attempts += 1
                ok, result, model_id = self.call(st, info["id"], client_api, body, ir)
                if ok:
                    return self._render(client_api, stream, result, model_id)
                status, message = result
                if classify_refusal(message) == "dead" or status in (400, 401, 403, 404):
                    st.dead = True
                excluded.add(st.id)
                continue
            excluded.add(st.id)
        return self._error(client_api, last_status)

    def _render(self, client_api, stream, ans, model_id):
        if client_api == "openai":
            if stream:
                return 200, "sse", render_openai_sse(ans, model_id)
            return 200, "json", json.dumps(render_openai_json(ans, model_id))
        if stream:
            return 200, "sse", render_anthropic_sse(ans, model_id)
        return 200, "json", json.dumps(render_anthropic_json(ans, model_id))

    def _error(self, client_api, status):
        status = status if 400 <= status <= 599 else 502
        if client_api == "openai":
            payload = {"error": {"message": "no free model could serve this request",
                                 "type": "api_error", "param": None, "code": "no_capacity"}}
        else:
            payload = {"type": "error", "error": {"type": "api_error",
                                                  "message": "no free model could serve this request"}}
        return status, "json", json.dumps(payload)


# --------------------------------------------------------------------------
# HTTP server
# --------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    world = None
    router = None
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # keep stdout clean
        pass

    def _send(self, status, content_type, text):
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.split("?")[0] == "/v1/models":
            ids = self.world.free_model_ids()
            if self.headers.get("x-api-key") is not None:
                payload = {"data": [{"type": "model", "id": mid, "display_name": mid,
                                     "created_at": CREATED_AT} for mid, _ in ids],
                           "has_more": False}
            else:
                payload = {"object": "list", "data": [
                    {"id": mid, "object": "model", "created": CREATED, "owned_by": owner}
                    for mid, owner in ids]}
            self._send(200, "application/json", json.dumps(payload))
            return
        self._send(404, "application/json", json.dumps({"error": {"message": "not found"}}))

    def do_POST(self):
        path = self.path.split("?")[0]
        api = None
        if path == "/v1/chat/completions":
            api = "openai"
        elif path == "/v1/messages":
            api = "anthropic"
        if api is None:
            self._send(404, "application/json", json.dumps({"error": {"message": "not found"}}))
            return
        try:
            length = int(self.headers.get("content-length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b"{}"
        body = _try_json(raw)
        if not isinstance(body, dict):
            body = {}
        status, kind, payload = self.router.serve(api, body)
        if kind == "sse":
            self._send(status, "text/event-stream", payload)
        else:
            self._send(status, "application/json", payload)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def load_world():
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE", "")
    env_url = os.environ.get("MAKEFUN_ENV_URL", "")
    text = ""
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as f:
            text = f.read()
    brief = Brief(text)
    world = World(brief, env_url)
    world.load_instance()
    world.build()
    return world


def main(argv):
    if "--list" in argv:
        if not os.environ.get("MAKEFUN_PROMPT_FILE"):
            print(json.dumps({"providers": []}))
            return 0
        world = load_world()
        sys.stdout.write(world.catalog_text + "\n")
        return 0

    world = load_world()
    router = Router(world)
    Handler.world = world
    Handler.router = router
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log("listening on 127.0.0.1:%d" % port)

    env_url = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    done = threading.Event()

    def on_signal(_signum, _frame):
        done.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    if env_url:
        # The catalog goes into the register body as the exact bytes --list
        # prints (world.catalog_text), so the two are identical by
        # construction.  Registration is retried until it succeeds: the
        # driver only starts on a successful register, so polling /work
        # before that would wait forever.
        register_body = ('{"protocol_version":%d,"port":%d,"catalog":%s}'
                         % (PROTOCOL_VERSION, port, world.catalog_text)).encode("utf-8")
        registered = False
        while not done.is_set():
            if not registered:
                status, payload, _ = http_raw("POST", env_url + "/register",
                                              register_body, timeout=5.0)
                if status == 200 and isinstance(payload, dict) and payload.get("ok"):
                    registered = True
                    log("registered; requests:", payload.get("requests"))
                else:
                    done.wait(0.3)
                continue
            status, payload, _ = http_json("GET", env_url + "/work", timeout=5.0)
            if status == 200 and isinstance(payload, dict) and payload.get("done"):
                break
            done.wait(0.1)
    else:
        done.wait()

    server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
