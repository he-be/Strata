"""serve/grammar.py - response_format json_schema -> the byte-level DFA the engine constrains decoding with.

llama-server turns a request's JSON schema into a grammar and only lets the model pick tokens the grammar allows;
this is the same for Strata.  The schema becomes an NFA over BYTES (so multi-byte characters split across tokens
need nothing special), then a DFA by subset construction, written to a file the engine reads
(src/program/token_grammar.hpp has the format and how the DFA meets the vocabulary).

The JSON is the compact-or-indented form models write: whitespace between the structural characters (llama.cpp's
`space` rule, so a model cannot pad forever), object members in the schema's order (llama.cpp's order too),
no members beyond `properties`.  A schema keyword this compiler does not implement is an error (HTTP 400), never
silently ignored - ignoring it is exactly the behavior this module exists to remove.

    compile_schema(schema) -> Dfa
    GrammarFiles(dir, tokenizer, eos_ids, think_end_id).path(schema, thinking) -> the file for a request
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

WS_INDENT = 20
MAX_STATES = 200_000

# keywords that only describe (they cannot change which JSON is valid)
_NOTES = {"title", "description", "$comment", "examples", "default", "$schema", "$id", "deprecated", "readOnly",
          "writeOnly"}


class SchemaError(ValueError):
    pass


class Nfa:
    def __init__(self):
        self.edges: list[dict[int, set[int]]] = []
        self.eps: list[set[int]] = []

    def new(self) -> int:
        self.edges.append({})
        self.eps.append(set())
        if len(self.edges) > MAX_STATES:
            raise SchemaError("the schema is too large to compile")
        return len(self.edges) - 1

    def byte(self, a: int, b: int, byte_set) -> None:
        for c in byte_set:
            self.edges[a].setdefault(c, set()).add(b)

    def link(self, a: int, b: int) -> None:
        self.eps[a].add(b)


class Builder:
    """Fragments are (start, end) state pairs of one NFA."""

    def __init__(self, root: dict):
        self.n = Nfa()
        self.root = root
        self.depth = 0

    # ---- primitives
    def lit(self, data: bytes) -> tuple[int, int]:
        s = cur = self.n.new()
        for c in data:
            nxt = self.n.new()
            self.n.byte(cur, nxt, (c,))
            cur = nxt
        return s, cur

    def seq(self, *frags) -> tuple[int, int]:
        s, e = frags[0]
        for a, b in frags[1:]:
            self.n.link(e, a)
            e = b
        return s, e

    def alt(self, frags) -> tuple[int, int]:
        s, e = self.n.new(), self.n.new()
        for a, b in frags:
            self.n.link(s, a)
            self.n.link(b, e)
        return s, e

    def ws(self) -> tuple[int, int]:
        """llama.cpp's `space` (json-schema-to-grammar): nothing, one space, or 1-2 newlines and up to 20 of indent."""
        s, e = self.n.new(), self.n.new()
        self.n.link(s, e)
        self.n.byte(s, e, b" ")
        nl1, nl2 = self.n.new(), self.n.new()
        self.n.byte(s, nl1, b"\n")
        self.n.byte(nl1, nl2, b"\n")
        cur = self.n.new()
        self.n.link(nl1, cur)
        self.n.link(nl2, cur)
        self.n.link(cur, e)
        for _ in range(WS_INDENT):
            nxt = self.n.new()
            self.n.byte(cur, nxt, b" \t")
            self.n.link(nxt, e)
            cur = nxt
        return s, e

    def string(self) -> tuple[int, int]:
        """A JSON string: no raw control characters, the JSON escapes, well-formed UTF-8 leads/continuations."""
        s, body, e = self.n.new(), self.n.new(), self.n.new()
        self.n.byte(s, body, (0x22,))
        self.n.byte(body, e, (0x22,))
        plain = [c for c in range(0x20, 0x7F) if c not in (0x22, 0x5C)]
        self.n.byte(body, body, plain)
        cont = range(0x80, 0xC0)
        for lead, k in ((range(0xC2, 0xE0), 1), (range(0xE0, 0xF0), 2), (range(0xF0, 0xF5), 3)):
            cur = self.n.new()
            self.n.byte(body, cur, lead)
            for _ in range(k - 1):
                nxt = self.n.new()
                self.n.byte(cur, nxt, cont)
                cur = nxt
            self.n.byte(cur, body, cont)
        esc = self.n.new()
        self.n.byte(body, esc, (0x5C,))
        self.n.byte(esc, body, b'"\\/bfnrt')
        cur = self.n.new()
        self.n.byte(esc, cur, b"u")
        hexd = b"0123456789abcdefABCDEF"
        for i in range(4):
            nxt = body if i == 3 else self.n.new()
            self.n.byte(cur, nxt, hexd)
            cur = nxt
        return s, e

    def digits(self, at_least: int) -> tuple[int, int]:
        s = cur = self.n.new()
        for _ in range(at_least):
            nxt = self.n.new()
            self.n.byte(cur, nxt, b"0123456789")
            cur = nxt
        self.n.byte(cur, cur, b"0123456789")
        return s, cur

    def integer(self) -> tuple[int, int]:
        minus = self.alt([self.lit(b""), self.lit(b"-")])
        body = self.alt([self.lit(b"0"), self.seq(self._range(b"123456789"), self.digits(0))])
        return self.seq(minus, body)

    def _range(self, chars: bytes) -> tuple[int, int]:
        s, e = self.n.new(), self.n.new()
        self.n.byte(s, e, chars)
        return s, e

    def number(self) -> tuple[int, int]:
        frac = self.alt([self.lit(b""), self.seq(self.lit(b"."), self.digits(1))])
        exp = self.alt([self.lit(b""), self.seq(self._range(b"eE"), self.alt([self.lit(b""), self._range(b"+-")]),
                                                  self.digits(1))])
        return self.seq(self.integer(), frac, exp)

    # ---- schema
    def resolve(self, ref: str) -> dict:
        if not ref.startswith("#/"):
            raise SchemaError(f"$ref {ref!r}: only local references (#/...) are supported")
        node = self.root
        for part in ref[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(node, dict) or part not in node:
                raise SchemaError(f"$ref {ref!r} does not resolve")
            node = node[part]
        return node

    def value(self, sch) -> tuple[int, int]:
        self.depth += 1
        if self.depth > 32:
            raise SchemaError("the schema nests deeper than 32 levels (a recursive $ref?)")
        try:
            return self._value(sch)
        finally:
            self.depth -= 1

    def _value(self, sch) -> tuple[int, int]:
        if sch is True or sch == {}:
            raise SchemaError("an unconstrained value ({} / true) is not supported: give it a type")
        if not isinstance(sch, dict):
            raise SchemaError(f"not a schema: {sch!r}")
        sch = {k: v for k, v in sch.items() if k not in _NOTES}
        if "$ref" in sch:
            rest = {k: v for k, v in sch.items() if k != "$ref"}
            if set(rest) - {"$defs", "definitions"}:
                raise SchemaError("$ref next to other keywords is not supported")
            return self.value(self.resolve(sch["$ref"]))
        sch.pop("$defs", None)
        sch.pop("definitions", None)
        for key in ("anyOf", "oneOf"):
            if key in sch:
                if len(sch) != 1:
                    raise SchemaError(f"{key} next to other keywords is not supported")
                return self.alt([self.value(x) for x in sch[key]])
        if "allOf" in sch:
            if len(sch) != 1 or len(sch["allOf"]) != 1:
                raise SchemaError("allOf is supported with one member only")
            return self.value(sch["allOf"][0])
        if "const" in sch:
            return self.lit(json.dumps(sch["const"], ensure_ascii=False).encode())
        if "enum" in sch:
            extra = set(sch) - {"enum", "type"}
            if extra:
                raise SchemaError(f"enum with {sorted(extra)} is not supported")
            return self.alt([self.lit(json.dumps(v, ensure_ascii=False).encode()) for v in sch["enum"]])
        t = sch.get("type")
        if isinstance(t, list):
            return self.alt([self.value({**sch, "type": x}) for x in t])
        known = {"type"}
        if t == "string":
            out = self.string()
        elif t == "integer":
            out = self.integer()
        elif t == "number":
            out = self.number()
        elif t == "boolean":
            out = self.alt([self.lit(b"true"), self.lit(b"false")])
        elif t == "null":
            out = self.lit(b"null")
        elif t == "array":
            known |= {"items", "minItems", "maxItems"}
            out = self.array(sch)
        elif t == "object":
            known |= {"properties", "required", "additionalProperties"}
            out = self.obj(sch)
        else:
            raise SchemaError(f"type {t!r} is not supported" if t else "a schema without a type is not supported")
        extra = set(sch) - known
        if extra:
            raise SchemaError(f"{sorted(extra)} on a {t} is not supported")
        return out

    def array(self, sch) -> tuple[int, int]:
        items = sch.get("items")
        if not isinstance(items, dict):
            raise SchemaError("an array needs one items schema")
        lo, hi = int(sch.get("minItems", 0)), sch.get("maxItems")
        hi = None if hi is None else int(hi)
        if hi is not None and hi < lo:
            raise SchemaError("maxItems < minItems")
        sep = lambda: self.seq(self.ws(), self.lit(b","), self.ws())  # noqa: E731
        s, e = self.lit(b"[")
        start, e = self.seq((s, e), self.ws())
        end = self.n.new()
        if lo == 0:
            self.n.link(e, end)
        cur = e
        for i in range(max(lo, 1) if hi is None else hi):
            a, b = self.value(items) if i == 0 else self.seq(sep(), self.value(items))
            self.n.link(cur, a)
            cur = b
            if i + 1 >= lo:
                self.n.link(cur, end)
        if hi is None:                           # more items: loop back through a separator
            a, b = self.seq(sep(), self.value(items))
            self.n.link(cur, a)
            self.n.link(b, a)
            self.n.link(b, end)
        return self.seq((start, end), self.ws(), self.lit(b"]"))

    def obj(self, sch) -> tuple[int, int]:
        props = sch.get("properties") or {}
        if sch.get("additionalProperties") not in (None, False):
            raise SchemaError("additionalProperties other than false is not supported")
        required = set(sch.get("required") or [])
        if required - set(props):
            raise SchemaError(f"required names unknown properties: {sorted(required - set(props))}")
        start, first = self.seq(self.lit(b"{"), self.ws())
        # two lanes: `none` (no member written yet) and `some` (a comma goes before the next member)
        none, some = first, self.n.new()
        for name, sub in props.items():
            member = lambda: self.seq(self.lit(json.dumps(name, ensure_ascii=False).encode()), self.ws(),  # noqa: E731
                                      self.lit(b":"), self.ws(), self.value(sub))
            n2, s2 = self.n.new(), self.n.new()
            a, b = member()
            self.n.link(none, a)
            self.n.link(b, s2)
            a, b = self.seq(self.ws(), self.lit(b","), self.ws(), member())
            self.n.link(some, a)
            self.n.link(b, s2)
            if name not in required:
                self.n.link(none, n2)
                self.n.link(some, s2)
            none, some = n2, s2
        end = self.n.new()
        self.n.link(none, end)
        self.n.link(some, end)
        return self.seq((start, end), self.ws(), self.lit(b"}"))


class Dfa:
    def __init__(self, table: list[list[int]], accept: list[bool], start: int):
        self.table, self.accept, self.start = table, accept, start

    def run(self, data: bytes) -> int:
        s = self.start
        for c in data:
            s = self.table[s][c]
            if s < 0:
                return -1
        return s


def compile_schema(schema) -> Dfa:
    b = Builder(schema if isinstance(schema, dict) else {})
    s, e = b.seq(b.value(schema), b.ws())   # llama.cpp: no space before the root value
    n = b.n

    def closure(states) -> frozenset:
        stack, seen = list(states), set(states)
        while stack:
            for t in n.eps[stack.pop()]:
                if t not in seen:
                    seen.add(t)
                    stack.append(t)
        return frozenset(seen)

    start = closure({s})
    index = {start: 0}
    order = [start]
    table: list[list[int]] = []
    accept: list[bool] = []
    i = 0
    while i < len(order):
        cur = order[i]
        accept.append(e in cur)
        moves: dict[int, set[int]] = {}
        for q in cur:
            for c, ts in n.edges[q].items():
                moves.setdefault(c, set()).update(ts)
        row = [-1] * 256
        for c, ts in moves.items():
            d = closure(ts)
            j = index.get(d)
            if j is None:
                j = index[d] = len(order)
                order.append(d)
                if len(order) > MAX_STATES:
                    raise SchemaError("the schema is too large to compile")
            row[c] = j
        table.append(row)
        i += 1
    return Dfa(table, accept, 0)


def _token_bytes(tokenizer, i: int) -> bytes:
    """A token's raw bytes: the tokenizer's own `token_bytes` (engine 0.1.2x+), else its byte-level BPE string
    mapped back through GPT-2's byte <-> unicode table (older tokenizers only have the strings)."""
    f = getattr(tokenizer, "token_bytes", None)
    if f is not None:
        return f(i)
    from strata_tokenizer import bytes_to_unicode
    inv = _token_bytes.__dict__.setdefault("inv", {v: k for k, v in bytes_to_unicode().items()})
    return bytes(inv[ch] for ch in tokenizer.tokens[i])


class GrammarFiles:
    """Grammar files for the engine, one per (schema, thinking), named by content so the engine's cache by path is
    exact; the vocabulary file is written once."""

    def __init__(self, directory: Path, tokenizer, eos_ids, think_end_id: int | None):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.eos = [int(i) for i in eos_ids]
        self.think_end = -1 if think_end_id is None else int(think_end_id)
        self.vocab_path = self.dir / "vocab.bin"
        types = getattr(tokenizer, "token_types", None) or []
        out = bytearray(b"SVOC" + struct.pack("<II", 1, len(tokenizer.tokens)))
        for i in range(len(tokenizer.tokens)):
            special = i < len(types) and types[i] in (3, 4)
            data = b""
            if not special:
                try:
                    data = _token_bytes(tokenizer, i)
                except (KeyError, IndexError):
                    special = True
            out += struct.pack("<BH", int(special), len(data)) + data
        tmp = self.vocab_path.with_suffix(".tmp")
        tmp.write_bytes(out)
        os.replace(tmp, self.vocab_path)
        self.cache: dict[str, Path] = {}

    def path(self, schema, thinking: bool) -> Path:
        key = hashlib.sha256(json.dumps([schema, thinking, self.eos, self.think_end], sort_keys=True,
                                        ensure_ascii=False).encode()).hexdigest()[:24]
        p = self.cache.get(key)
        if p is not None and p.exists():
            return p
        dfa = compile_schema(schema)
        vp = str(self.vocab_path).encode()
        out = bytearray(b"SGRM")
        out += struct.pack("<IIii", 1, len(dfa.table), dfa.start, self.think_end if thinking else -1)
        out += struct.pack("<I", len(self.eos)) + b"".join(struct.pack("<i", x) for x in self.eos)
        out += struct.pack("<I", len(vp)) + vp
        out += bytes(int(a) for a in dfa.accept)
        out += b"".join(struct.pack("<256i", *row) for row in dfa.table)
        p = self.dir / f"g-{key}.bin"
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(out)
        os.replace(tmp, p)
        self.cache[key] = p
        return p


def schema_of(response_format) -> dict | None:
    """The schema a request's response_format asks for, None for plain text; SchemaError for what is unsupported."""
    if response_format is None:
        return None
    if not isinstance(response_format, dict):
        raise SchemaError("response_format must be an object")
    kind = response_format.get("type")
    if kind in (None, "text"):
        return None
    if kind == "json_schema":
        js = response_format.get("json_schema")
        sch = js.get("schema") if isinstance(js, dict) else None
        if not isinstance(sch, dict):
            raise SchemaError("response_format.json_schema.schema is missing")
        return sch
    if kind == "json_object":
        s = response_format.get("schema")
        if isinstance(s, dict):
            return s
        raise SchemaError("response_format json_object without a schema is not supported; use json_schema")
    raise SchemaError(f"response_format type {kind!r} is not supported")
