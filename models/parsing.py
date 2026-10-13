"""Parse only the three declared surface grammars; never read answer metadata."""

import re

from synthetic_dataset_generator.logic import Rule
from .representations import format_input

TOKEN = re.compile(r"p[0-9]{3}|[A-Za-z]+|->|[~&|(),;:.]")
ATOM = re.compile(r"p[0-9]{3}\Z")


class Parser:
    def __init__(self, text):
        self.tokens = []
        offset = 0
        for match in TOKEN.finditer(text):
            if text[offset:match.start()].strip():
                raise ValueError("Unsupported input syntax")
            self.tokens.append(match.group())
            offset = match.end()
        if text[offset:].strip():
            raise ValueError("Unsupported trailing syntax")
        self.index = 0

    def peek(self):
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def take(self, expected=None):
        token = self.peek()
        if token is None or (expected is not None and token != expected):
            raise ValueError(f"Expected {expected!r}, got {token!r}")
        self.index += 1
        return token

    def literal(self, negation="~"):
        negative = self.peek() == negation
        if negative:
            self.take()
        atom = self.take()
        if not ATOM.fullmatch(atom):
            raise ValueError(f"Expected atom, got {atom!r}")
        return ("~" if negative else "") + atom

    def expression(self, english=False):
        """AND binds more tightly than OR; parentheses preserve groupings."""
        and_token, or_token = ("and", "or") if english else ("&", "|")

        def term():
            if self.peek() == "(":
                self.take("(")
                value = self.expression(english)
                self.take(")")
                return value
            return ((self.literal("not" if english else "~"),),)

        def conjunction():
            value = term()
            while self.peek() == and_token:
                self.take()
                value = tuple(a+b for a in value for b in term())
                self.bounded(value)
            return value

        value = conjunction()
        while self.peek() == or_token:
            self.take()
            value += conjunction()
            self.bounded(value)
        return value

    @staticmethod
    def bounded(clauses):
        if not 1 <= len(clauses) <= 4 or any(not 1 <= len(c) <= 3 for c in clauses):
            raise ValueError("Expression exceeds controlled-fragment bounds")

    def function(self):
        name = self.take()
        if ATOM.fullmatch(name):
            return ((name,),)
        self.take("(")
        if name == "not":
            atom = self.take()
            if not ATOM.fullmatch(atom):
                raise ValueError("not() must contain exactly one atom")
            self.take(")")
            return (("~" + atom,),)
        if name not in ("and", "or"):
            raise ValueError(f"Unsupported function: {name}")
        arguments = [self.function()]
        while self.peek() == ",":
            self.take(",")
            arguments.append(self.function())
        self.take(")")
        if len(arguments) < 2:
            raise ValueError("AND/OR require at least two operands")
        value = arguments[0]
        for other in arguments[1:]:
            value = value+other if name == "or" else tuple(a+b for a in value for b in other)
            self.bounded(value)
        return value

    def function_literal(self):
        clauses = self.function()
        if len(clauses) != 1 or len(clauses[0]) != 1:
            raise ValueError("Facts, conclusions and queries must be literals")
        return clauses[0][0]

    def finish(self):
        if self.peek() is not None:
            raise ValueError(f"Unexpected trailing token {self.peek()!r}")


def parse(text, representation):
    parser = Parser(text)
    facts, rules = [], []
    if representation == "symbolic":
        parser.take("FACTS")
        parser.take(":")
        if parser.peek() != ".":
            facts.append(parser.literal())
            while parser.peek() == ";":
                parser.take()
                facts.append(parser.literal())
        parser.take(".")
        parser.take("RULES")
        parser.take(":")
        if parser.peek() != ".":
            while True:
                premise = parser.expression()
                parser.take("->")
                rules.append(Rule(premise, parser.literal()))
                if parser.peek() != ";":
                    break
                parser.take()
        parser.take(".")
        parser.take("QUERY")
        parser.take(":")
        query = parser.literal()
        parser.take(".")
    elif representation == "function":
        parser.take("facts")
        parser.take("(")
        if parser.peek() != ")":
            facts.append(parser.function_literal())
            while parser.peek() == ",":
                parser.take()
                facts.append(parser.function_literal())
        parser.take(")")
        parser.take(";")
        parser.take("rules")
        parser.take("(")
        if parser.peek() != ")":
            while True:
                parser.take("implies")
                parser.take("(")
                premise = parser.function()
                parser.take(",")
                rules.append(Rule(premise, parser.function_literal()))
                parser.take(")")
                if parser.peek() != ";":
                    break
                parser.take()
        parser.take(")")
        parser.take(";")
        parser.take("query")
        parser.take("(")
        query = parser.function_literal()
        parser.take(")")
    elif representation == "english":
        parser.take("Known")
        parser.take(":")
        if parser.peek() != ".":
            facts.append(parser.literal("not"))
            while parser.peek() == ",":
                parser.take()
                facts.append(parser.literal("not"))
        parser.take(".")
        while parser.peek() == "If":
            parser.take("If")
            premise = parser.expression(english=True)
            parser.take(",")
            parser.take("then")
            rules.append(Rule(premise, parser.literal("not")))
            parser.take(".")
        parser.take("Query")
        parser.take(":")
        query = parser.literal("not")
        parser.take(".")
    else:
        raise ValueError(f"Unknown representation {representation}")
    parser.finish()
    for rule in rules:
        parser.bounded(rule.clauses)
    return {"metadata": {"facts": facts, "rules": [r.to_dict() for r in rules], "query": query}}


def normalize(text, source_representation, target_representation="symbolic"):
    return format_input(parse(text, source_representation), target_representation)
