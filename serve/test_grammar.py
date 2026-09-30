"""serve/grammar.py: the schema -> byte DFA compiler.

    python -m unittest serve.test_grammar -v
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.grammar import SchemaError, compile_schema, schema_of  # noqa: E402

GATE = {"type": "object", "additionalProperties": False, "required": ["checks"], "properties": {
    "checks": {"type": "array", "minItems": 2, "maxItems": 2, "items": {
        "type": "object", "additionalProperties": False, "required": ["name", "evidence", "ok"], "properties": {
            "name": {"type": "string", "enum": ["絵柄", "主役"]},
            "evidence": {"type": "string"},
            "ok": {"type": "string", "enum": ["はい", "いいえ"]}}}}}}


def accepts(dfa, text: str) -> bool:
    s = dfa.run(text.encode())
    return s >= 0 and dfa.accept[s]


class Grammar(unittest.TestCase):
    def test_schema_output_passes(self):
        d = compile_schema(GATE)
        obj = {"checks": [{"name": "絵柄", "evidence": "線が \"細い\"\\ ✓", "ok": "はい"},
                          {"name": "主役", "evidence": "", "ok": "いいえ"}]}
        self.assertTrue(accepts(d, json.dumps(obj, ensure_ascii=False)))
        self.assertTrue(accepts(d, json.dumps(obj, ensure_ascii=False, indent=2)))
        # \\u escapes in a free string (an enum is its literal, as in llama.cpp)
        self.assertTrue(accepts(d, json.dumps(obj, ensure_ascii=False).replace("✓", "\\u2713")))

    def test_off_schema_output_dies(self):
        d = compile_schema(GATE)
        good = '{"checks": [{"name": "絵柄", "evidence": "x", "ok": "はい"}, {"name": "主役", "evidence": "y", "ok": "はい"}]}'
        self.assertFalse(accepts(d, "```json\n" + good))                # a code fence
        self.assertFalse(accepts(d, good.replace('"ok": "はい"}]', '"ok": "たぶん"}]')))   # outside the enum
        self.assertFalse(accepts(d, good.replace(', {"name": "主役", "evidence": "y", "ok": "はい"}', "")))  # minItems
        self.assertFalse(accepts(d, good[:-1] + ', "more": 1}'))         # additionalProperties
        self.assertFalse(accepts(d, good.replace('"evidence": "x", ', "")))   # a required member
        self.assertFalse(accepts(d, good.replace('"x"', '"a\nb"')))       # a raw newline in a string
        self.assertFalse(accepts(d, " " + good))                          # llama.cpp: nothing before the root
        self.assertEqual(d.run(good[:-1].encode()) >= 0, True)           # a prefix stays alive ...
        self.assertFalse(accepts(d, good[:-1]))                          # ... but is not the end

    def test_optional_members_and_numbers(self):
        d = compile_schema({"type": "object", "required": ["b"], "properties": {
            "a": {"type": "integer"}, "b": {"type": "number"}, "c": {"type": ["boolean", "null"]}}})
        for ok in ('{"b": 1}', '{"a": -3, "b": 2.5e-3}', '{"b": 0, "c": null}', '{"a": 1, "b": 1, "c": true}'):
            self.assertTrue(accepts(d, ok), ok)
        for bad in ('{"a": 1}', '{"b": 01}', '{"b": 1, "a": 1}', '{, "b": 1}'):
            self.assertFalse(accepts(d, bad), bad)

    def test_unsupported_is_an_error(self):
        for sch in ({"type": "string", "pattern": "a+"}, {"type": "object", "additionalProperties": True},
                    {}, {"type": "integer", "minimum": 0}):
            with self.assertRaises(SchemaError):
                compile_schema(sch)
        with self.assertRaises(SchemaError):
            schema_of({"type": "json_object"})
        self.assertIsNone(schema_of({"type": "text"}))


if __name__ == "__main__":
    unittest.main()
