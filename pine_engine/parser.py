"""Leser Pine Script v4–v6 til et syntakstre.

Støtter variabler (``var``/``varip``, typer, ``:=``, ``+=``), tupler, ``if``/``else if``/``else``,
``for … to … by``, ``for … in``, ``while``, ``switch``, ``if``/``switch`` som uttrykk, egne
funksjoner (én linje eller blokk), historikk ``x[n]``, ternær, navngitte argumenter og generiske
typer (``array.new<float>``). Linjer fortsetter når parentesen ikke er lukket eller neste linje er
rykket inn med et antall mellomrom som ikke er delelig med 4 (Pine-regelen).
"""
import re
from dataclasses import dataclass, field


class PineError(Exception):
    """Feil i skriptet (syntaks, ukjent funksjon eller kjøring), med linjenummer når det er kjent."""

    def __init__(self, message, line=None):
        self.line = line
        super().__init__(f"Linje {line}: {message}" if line else message)


# ---------------------------------------------------------------- syntakstre

@dataclass
class Num:
    value: float


@dataclass
class Str:
    value: str


@dataclass
class Bool:
    value: bool


@dataclass
class Na:
    pass


@dataclass
class Color:
    value: str


@dataclass
class Name:
    id: str
    line: int = 0


@dataclass
class Attr:
    obj: object
    attr: str
    line: int = 0


@dataclass
class Call:
    func: object
    args: list
    kwargs: list
    line: int = 0
    type_args: list = field(default_factory=list)


@dataclass
class Index:
    expr: object
    offset: object
    line: int = 0


@dataclass
class BinOp:
    op: str
    left: object
    right: object


@dataclass
class UnaryOp:
    op: str
    operand: object


@dataclass
class Ternary:
    cond: object
    a: object
    b: object


@dataclass
class TupleLit:
    items: list
    line: int = 0


@dataclass
class BlockExpr:
    """``if``/``switch`` brukt som uttrykk; verdien er siste uttrykk i blokken som kjøres."""
    stmt: object


@dataclass
class Decl:
    name: str
    expr: object
    mode: str | None
    typ: str | None
    line: int = 0


@dataclass
class TupleDecl:
    names: list
    expr: object
    reassign: bool
    line: int = 0


@dataclass
class Assign:
    name: str
    op: str
    expr: object
    line: int = 0


@dataclass
class ExprStmt:
    expr: object
    line: int = 0


@dataclass
class If:
    cond: object
    body: list
    elifs: list
    orelse: list | None
    line: int = 0


@dataclass
class For:
    var: str
    start: object
    end: object
    step: object
    body: list
    line: int = 0


@dataclass
class ForIn:
    names: list
    iterable: object
    body: list
    line: int = 0


@dataclass
class While:
    cond: object
    body: list
    line: int = 0


@dataclass
class Switch:
    subject: object
    cases: list          # [(uttrykk eller None for standard, kropp)]
    line: int = 0


@dataclass
class Break:
    line: int = 0


@dataclass
class Continue:
    line: int = 0


@dataclass
class FuncDef:
    name: str
    params: list         # [(navn, standardverdi eller None)]
    body: list
    line: int = 0
    method_of: str | None = None     # mottakertype for ``method`` (f.eks. «ReactionLevel» eller «array»)


@dataclass
class TypeDef:
    """``type Navn`` med felt: [(feltnavn, standarduttrykk eller None)]."""
    name: str
    fields: list
    line: int = 0


@dataclass
class EnumDef:
    """``enum Navn`` med medlemmer: [(navn, tittel eller None)]."""
    name: str
    members: list
    line: int = 0


@dataclass
class FieldAssign:
    """``objekt.felt := verdi`` (også ``+=``, ``-=`` osv.)."""
    obj: object
    field: str
    op: str
    expr: object
    line: int = 0


@dataclass
class Program:
    version: int
    body: list
    source_lines: int


# ---------------------------------------------------------------- tegn til tokens

TYPE_WORDS = {"int", "float", "bool", "string", "color", "line", "label", "box", "table", "linefill",
              "polyline", "array", "matrix", "map"}
QUALIFIERS = {"series", "simple", "const", "input"}
KEYWORDS = {"and", "or", "not", "if", "else", "for", "to", "by", "in", "while", "switch", "var", "varip",
            "true", "false", "break", "continue", "import", "export", "method", "type"}
BINARY_TAIL = {"?", ":", ",", "and", "or", "+", "-", "*", "/", "%", "==", "!=", "<", ">", "<=", ">=", "=", ":="}
OPERATORS = (":=", "+=", "-=", "*=", "/=", "%=", "==", "!=", "<=", ">=", "=>", "<", ">", "=", "+", "-", "*",
             "/", "%", "?", ":", ",", "(", ")", "[", "]", ".")
TOKEN = re.compile(r"""
    (?P<ws>[ \t]+)
  | (?P<comment>//.*)
  | (?P<number>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<string>"(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')
  | (?P<color>\#[0-9a-fA-F]{8}|\#[0-9a-fA-F]{6})
  | (?P<name>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<op>:=|\+=|-=|\*=|/=|%=|==|!=|<=|>=|=>|[<>=+\-*/%?:,()\[\].])
""", re.VERBOSE)
ESCAPES = {"n": "\n", "t": "\t", "\\": "\\", "'": "'", '"': '"', "r": "\r"}


@dataclass
class Tok:
    kind: str
    value: str
    line: int


@dataclass
class Logical:
    indent: int
    tokens: list
    line: int
    depth: int = 0


def _unescape(text):
    out, k = [], 0
    while k < len(text):
        ch = text[k]
        if ch == "\\" and k + 1 < len(text):
            out.append(ESCAPES.get(text[k + 1], text[k + 1]))
            k += 2
        else:
            out.append(ch)
            k += 1
    return "".join(out)


def tokenize_line(raw, line):
    tokens, pos = [], 0
    while pos < len(raw):
        match = TOKEN.match(raw, pos)
        if not match:
            raise PineError(f"ukjent tegn «{raw[pos]}»", line)
        pos = match.end()
        kind = match.lastgroup
        if kind == "ws":
            continue
        if kind == "comment":
            break
        value = match.group()
        if kind == "string":
            value = _unescape(value[1:-1])
        elif kind == "name" and value in KEYWORDS:
            kind = "kw"
        tokens.append(Tok(kind, value, line))
    return tokens


def _depth(tokens):
    return sum(1 for t in tokens if t.kind == "op" and t.value in "([") - \
        sum(1 for t in tokens if t.kind == "op" and t.value in ")]")


def _indent(raw):
    width = 0
    for ch in raw:
        if ch == " ":
            width += 1
        elif ch == "\t":
            width += 4
        else:
            break
    return width


def logical_lines(source):
    """Fysiske linjer → logiske linjer (innrykksnivå, tokens) etter Pines fortsettelsesregler."""
    out, current = [], None
    for number, raw in enumerate(source.replace("\r\n", "\n").replace("\r", "\n").split("\n"), 1):
        tokens = tokenize_line(raw, number)
        if not tokens:
            continue
        width = _indent(raw)
        if current is not None:
            tail = current.tokens[-1]
            starts_operator = tokens[0].value in ("?", ":", "and", "or", "+", "*", "/", "==", "!=", ")", "]") \
                and tokens[0].kind in ("op", "kw")
            if current.depth > 0 or width % 4 != 0 or \
                    (width > current.indent * 4 and (tail.value in BINARY_TAIL and tail.kind in ("op", "kw")
                                                    or starts_operator)):
                current.tokens.extend(tokens)
                current.depth += _depth(tokens)
                continue
        current = Logical(width // 4, tokens, number, _depth(tokens))
        out.append(current)
    if current is not None and current.depth > 0:
        raise PineError("parentesen lukkes aldri", current.line)
    return out


# ---------------------------------------------------------------- uttrykk

class Tokens:
    def __init__(self, tokens, line):
        self.t, self.k, self.line = tokens, 0, line

    def peek(self, offset=0):
        k = self.k + offset
        return self.t[k] if k < len(self.t) else None

    def at(self, value, offset=0):
        tok = self.peek(offset)
        return tok is not None and tok.value == value and tok.kind in ("op", "kw")

    def next(self):
        tok = self.peek()
        if tok is None:
            raise PineError("uttrykket slutter for tidlig", self.line)
        self.k += 1
        return tok

    def expect(self, value):
        tok = self.next()
        if tok.value != value or tok.kind not in ("op", "kw"):
            raise PineError(f"forventet «{value}», fikk «{tok.value}»", tok.line)
        return tok

    def done(self):
        return self.k >= len(self.t)


BINARY_LEVELS = [("or",), ("and",), ("==", "!="), ("<", ">", "<=", ">="), ("+", "-"), ("*", "/", "%")]


def parse_expr(ts):
    return _ternary(ts)


def _ternary(ts):
    cond = _binary(ts, 0)
    if ts.at("?"):
        ts.next()
        a = _ternary(ts)
        ts.expect(":")
        b = _ternary(ts)
        return Ternary(cond, a, b)
    return cond


def _binary(ts, level):
    if level == len(BINARY_LEVELS):
        return _unary(ts)
    left = _binary(ts, level + 1)
    while True:
        tok = ts.peek()
        if tok is not None and tok.kind in ("op", "kw") and tok.value in BINARY_LEVELS[level]:
            ts.next()
            left = BinOp(tok.value, left, _binary(ts, level + 1))
        else:
            return left


def _unary(ts):
    tok = ts.peek()
    if tok is not None and tok.kind in ("op", "kw") and tok.value in ("not", "-", "+"):
        ts.next()
        return UnaryOp(tok.value, _unary(ts))
    return _postfix(ts)


def _type_args(ts):
    """``<float>`` / ``<string, float>`` rett før en parentes, ellers None (da er ``<`` en sammenligning)."""
    if not ts.at("<"):
        return None
    k, depth, names = 1, 1, []
    while True:
        tok = ts.peek(k)
        if tok is None:
            return None
        if tok.kind == "name":
            names.append(tok.value)
        elif tok.value == "<":
            depth += 1
        elif tok.value == ">":
            depth -= 1
            if depth == 0:
                break
        elif tok.value not in (",", "."):
            return None
        k += 1
    if not ts.at("(", k + 1):
        return None
    ts.k += k + 1
    return names


def _postfix(ts):
    node = _primary(ts)
    while True:
        if ts.at("."):
            ts.next()
            tok = ts.next()
            if tok.kind not in ("name", "kw"):
                raise PineError(f"forventet navn etter «.», fikk «{tok.value}»", tok.line)
            node = Attr(node, tok.value, tok.line)
        elif ts.at("("):
            line = ts.next().line
            node = _call(ts, node, line, [])
        elif ts.at("[") :
            line = ts.next().line
            offset = parse_expr(ts)
            ts.expect("]")
            node = Index(node, offset, line)
        elif isinstance(node, (Name, Attr)) and ts.at("<"):
            types = _type_args(ts)
            if types is None:
                return node
            line = ts.next().line
            node = _call(ts, node, line, types)
        else:
            return node


def _call(ts, func, line, type_args):
    args, kwargs = [], []
    while not ts.at(")"):
        tok, after = ts.peek(), ts.peek(1)
        if tok is not None and tok.kind in ("name", "kw") and after is not None and after.kind == "op" and \
                after.value == "=":
            ts.k += 2
            kwargs.append((tok.value, parse_expr(ts)))
        else:
            if kwargs:
                raise PineError("posisjonsargument etter navngitt argument", line)
            args.append(parse_expr(ts))
        if not ts.at(")"):
            ts.expect(",")
    ts.expect(")")
    return Call(func, args, kwargs, line, type_args)


def _primary(ts):
    tok = ts.next()
    if tok.kind == "number":
        return Num(float(tok.value))
    if tok.kind == "string":
        return Str(tok.value)
    if tok.kind == "color":
        return Color(tok.value)
    if tok.kind == "kw" and tok.value in ("true", "false"):
        return Bool(tok.value == "true")
    if tok.kind == "name":
        return Na() if tok.value == "na" and not ts.at("(") else Name(tok.value, tok.line)
    if tok.kind == "op" and tok.value == "(":
        node = parse_expr(ts)
        ts.expect(")")
        return node
    if tok.kind == "op" and tok.value == "[":
        items = []
        while not ts.at("]"):
            items.append(parse_expr(ts))
            if not ts.at("]"):
                ts.expect(",")
        ts.expect("]")
        return TupleLit(items, tok.line)
    raise PineError(f"uventet «{tok.value}»", tok.line)


# ---------------------------------------------------------------- setninger

class Parser:
    def __init__(self, lines):
        self.lines, self.pos = lines, 0

    def block(self, level):
        body = []
        while self.pos < len(self.lines) and self.lines[self.pos].indent >= level:
            line = self.lines[self.pos]
            if line.indent > level:
                raise PineError("uventet innrykk (blokker må rykkes inn med 4 mellomrom eller tab)", line.line)
            body.append(self.statement(line, level))
        return body

    def child_block(self, level, line):
        if self.pos >= len(self.lines) or self.lines[self.pos].indent <= level:
            raise PineError("blokken under mangler innhold", line)
        return self.block(level + 1)

    def statement(self, logical, level):
        self.pos += 1
        toks, line = logical.tokens, logical.line
        first = toks[0]
        if first.kind == "kw":
            if first.value == "if":
                return self.if_stmt(toks[1:], level, line)
            if first.value == "for":
                return self.for_stmt(toks[1:], level, line)
            if first.value == "while":
                cond = self.expr(toks[1:], line)
                return While(cond, self.child_block(level, line), line)
            if first.value == "switch":
                return self.switch_stmt(toks[1:], level, line)
            if first.value == "break":
                return Break(line)
            if first.value == "continue":
                return Continue(line)
            if first.value == "export" and len(toks) > 1:
                self.pos -= 1
                return self.statement(Logical(logical.indent, toks[1:], line), level)
            if first.value == "type" and len(toks) == 2 and toks[1].kind == "name":
                return self.type_def(toks[1].value, level, line)
            if first.value == "method" and len(toks) > 1:
                func = self.funcdef(toks[1:], level, line, method=True)
                if func is None:
                    raise PineError("method må ha formen «method navn(Type this, ...) =>»", line)
                return func
            if first.value == "import":
                raise PineError("«import» av biblioteker støttes ikke ennå (lim inn bibliotekets funksjoner i "
                                "skriptet i stedet)", line)
        if first.kind == "name" and first.value == "enum" and len(toks) == 2 and toks[1].kind == "name":
            return self.enum_def(toks[1].value, level, line)
        func = self.funcdef(toks, level, line)
        if func is not None:
            return func
        if first.kind == "op" and first.value == "[":
            close = self._matching(toks, 0)
            if close + 1 < len(toks) and toks[close + 1].value in ("=", ":="):
                names = [t.value for t in toks[1:close] if t.kind == "name"]
                return TupleDecl(names, self.rhs(toks[close + 2:], level, line), toks[close + 1].value == ":=", line)
        decl = self.declaration(toks, level, line)
        if decl is not None:
            return decl
        if first.kind == "name" and len(toks) > 1 and toks[1].kind == "op" and \
                toks[1].value in (":=", "+=", "-=", "*=", "/=", "%="):
            return Assign(first.value, toks[1].value, self.rhs(toks[2:], level, line), line)
        field = self.field_assign(toks, level, line)
        if field is not None:
            return field
        if first.kind == "kw" and first.value in ("if", "switch"):
            return ExprStmt(self.rhs(toks, level, line), line)
        return ExprStmt(self.expr(toks, line), line)

    def expr(self, toks, line):
        if not toks:
            raise PineError("mangler uttrykk", line)
        ts = Tokens(toks, line)
        node = parse_expr(ts)
        if not ts.done():
            raise PineError(f"uventet «{ts.peek().value}»", ts.peek().line)
        return node

    def rhs(self, toks, level, line):
        """Høyre side av en tilordning: vanlig uttrykk eller ``if``/``switch`` med blokk."""
        if toks and toks[0].kind == "kw" and toks[0].value == "if":
            return BlockExpr(self.if_stmt(toks[1:], level, line))
        if toks and toks[0].kind == "kw" and toks[0].value == "switch":
            return BlockExpr(self.switch_stmt(toks[1:], level, line))
        return self.expr(toks, line)

    @staticmethod
    def _matching(toks, k):
        depth = 0
        for j in range(k, len(toks)):
            if toks[j].kind == "op" and toks[j].value in "([":
                depth += 1
            elif toks[j].kind == "op" and toks[j].value in ")]":
                depth -= 1
                if depth == 0:
                    return j
        raise PineError("parentesen lukkes aldri", toks[k].line)

    def funcdef(self, toks, level, line, method=False):
        if not (len(toks) >= 3 and toks[0].kind == "name" and toks[1].value == "(" and toks[1].kind == "op"):
            return None
        close = self._matching(toks, 1)
        if close + 1 >= len(toks) or toks[close + 1].value != "=>":
            return None
        params, types, k = [], [], 2
        while k < close:
            group = []
            while k < close and not (toks[k].value == "," and toks[k].kind == "op"):
                group.append(toks[k])
                k += 1
            k += 1
            if not group:
                continue
            default = None
            if any(t.value == "=" and t.kind == "op" for t in group):
                eq = next(j for j, t in enumerate(group) if t.value == "=" and t.kind == "op")
                default = self.expr(group[eq + 1:], line)
                group = group[:eq]
            names = [t for t in group if t.kind == "name"]   # navnet er siste ord; typeord står foran
            if not names:
                raise PineError("parameter uten navn", line)
            params.append((names[-1].value, default))
            kinds = [t.value for t in names[:-1] if t.value not in QUALIFIERS]
            types.append(kinds[0] if kinds else None)
        rest = toks[close + 2:]
        body = [ExprStmt(self.rhs(rest, level, line), line)] if rest else self.child_block(level, line)
        receiver = None
        if method:
            if not params or types[0] is None:
                raise PineError("første parameter i en method må ha en type (f.eks. «Level this»)", line)
            receiver = types[0]
        return FuncDef(toks[0].value, params, body, line, receiver)

    def type_def(self, name, level, line):
        fields = []
        for _ in range(10_000):
            if self.pos >= len(self.lines) or self.lines[self.pos].indent <= level:
                break
            item = self.lines[self.pos]
            self.pos += 1
            toks = [t for t in item.tokens if not (t.kind == "kw" and t.value == "varip")]
            eq = next((j for j, t in enumerate(toks) if t.kind == "op" and t.value == "="), None)
            head = toks if eq is None else toks[:eq]
            names = [t for t in head if t.kind in ("name", "kw")]
            if len(names) < 2:
                raise PineError("felt i en type må ha formen «type navn [= standard]»", item.line)
            default = self.expr(toks[eq + 1:], item.line) if eq is not None else None
            fields.append((names[-1].value, default))
        if not fields:
            raise PineError(f"typen «{name}» har ingen felt", line)
        return TypeDef(name, fields, line)

    def enum_def(self, name, level, line):
        members = []
        while self.pos < len(self.lines) and self.lines[self.pos].indent > level:
            item = self.lines[self.pos]
            self.pos += 1
            toks = item.tokens
            title = None
            if len(toks) >= 3 and toks[1].value == "=":
                title = self.expr(toks[2:], item.line)
            members.append((toks[0].value, title))
        return EnumDef(name, members, line)

    def field_assign(self, toks, level, line):
        """``a.b := x`` / ``arr.get(0).felt += 1``: venstresiden er et felt på et objekt."""
        depth = 0
        for j, t in enumerate(toks):
            if t.kind == "op" and t.value in "([":
                depth += 1
            elif t.kind == "op" and t.value in ")]":
                depth -= 1
            elif depth == 0 and t.kind == "op" and t.value in (":=", "+=", "-=", "*=", "/=", "%="):
                target = self.expr(toks[:j], line)
                if not isinstance(target, Attr):
                    return None
                return FieldAssign(target.obj, target.attr, t.value, self.rhs(toks[j + 1:], level, line), line)
        return None

    def declaration(self, toks, level, line):
        k, mode, typ = 0, None, None
        if toks[0].kind == "kw" and toks[0].value in ("var", "varip"):
            mode, k = toks[0].value, 1
        while k < len(toks) and toks[k].kind == "name" and toks[k].value in QUALIFIERS and \
                k + 1 < len(toks) and toks[k + 1].kind == "name":
            k += 1
        if k + 1 < len(toks) and toks[k].kind == "name" and (toks[k + 1].kind == "name" or
                                                              toks[k + 1].value in ("<", ".")):
            j = k + 1
            while j + 1 < len(toks) and toks[j].value == "." and toks[j + 1].kind == "name":
                j += 2
            if j < len(toks) and toks[j].value == "<":
                depth = 0
                while j < len(toks):
                    if toks[j].value == "<":
                        depth += 1
                    elif toks[j].value == ">":
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                j += 1
            if j + 1 < len(toks) and toks[j].kind == "name" and toks[j + 1].value == "=" and toks[j + 1].kind == "op":
                typ, k = toks[k].value, j
        if k + 1 < len(toks) and toks[k].kind == "name" and toks[k + 1].kind == "op" and toks[k + 1].value == "=":
            return Decl(toks[k].value, self.rhs(toks[k + 2:], level, line), mode, typ, line)
        if mode:
            raise PineError("«var» må etterfølges av et variabelnavn og «=»", line)
        return None

    def if_stmt(self, toks, level, line):
        cond = self.expr(toks, line)
        body = self.child_block(level, line)
        elifs, orelse = [], None
        while self.pos < len(self.lines) and self.lines[self.pos].indent == level:
            nxt = self.lines[self.pos]
            if not (nxt.tokens[0].kind == "kw" and nxt.tokens[0].value == "else"):
                break
            self.pos += 1
            if len(nxt.tokens) > 1 and nxt.tokens[1].value == "if":
                elifs.append((self.expr(nxt.tokens[2:], nxt.line), self.child_block(level, nxt.line)))
            else:
                if len(nxt.tokens) > 1:
                    raise PineError("uventet tekst etter «else»", nxt.line)
                orelse = self.child_block(level, nxt.line)
                break
        return If(cond, body, elifs, orelse, line)

    def for_stmt(self, toks, level, line):
        if toks and toks[0].value == "[":
            close = self._matching(toks, 0)
            names = [t.value for t in toks[1:close] if t.kind == "name"]
            if close + 1 >= len(toks) or toks[close + 1].value != "in":
                raise PineError("forventet «in» i for-løkke", line)
            return ForIn(names, self.expr(toks[close + 2:], line), self.child_block(level, line), line)
        if len(toks) >= 2 and toks[0].kind == "name" and toks[1].value == "in":
            return ForIn([toks[0].value], self.expr(toks[2:], line), self.child_block(level, line), line)
        if len(toks) < 4 or toks[0].kind == "kw" or toks[1].value != "=":
            raise PineError("for-løkken må ha formen «for i = a to b»", line)
        to = next((j for j, t in enumerate(toks) if t.kind == "kw" and t.value == "to"), None)
        if to is None:
            raise PineError("for-løkken mangler «to»", line)
        by = next((j for j, t in enumerate(toks) if t.kind == "kw" and t.value == "by"), None)
        start = self.expr(toks[2:to], line)
        end = self.expr(toks[to + 1:by] if by else toks[to + 1:], line)
        step = self.expr(toks[by + 1:], line) if by else None
        return For(toks[0].value, start, end, step, self.child_block(level, line), line)

    def switch_stmt(self, toks, level, line):
        subject = self.expr(toks, line) if toks else None
        if self.pos >= len(self.lines) or self.lines[self.pos].indent <= level:
            raise PineError("switch mangler tilfeller", line)
        cases = []
        while self.pos < len(self.lines) and self.lines[self.pos].indent == level + 1:
            case = self.lines[self.pos]
            self.pos += 1
            arrow = next((j for j, t in enumerate(case.tokens) if t.kind == "op" and t.value == "=>"), None)
            if arrow is None:
                raise PineError("switch-tilfellet mangler «=>»", case.line)
            left, right = case.tokens[:arrow], case.tokens[arrow + 1:]
            key = self.expr(left, case.line) if left else None
            body = [ExprStmt(self.rhs(right, level + 1, case.line), case.line)] if right else \
                self.child_block(level + 1, case.line)
            cases.append((key, body))
        return Switch(subject, cases, line)


VERSION = re.compile(r"//\s*@version\s*=\s*(\d+)")
EXPAND = re.compile(r"^\s*(Expand|Collapse|Vis mer|Vis mindre)\s*\(\d+\s*(lines?|linjer)\)\s*$", re.IGNORECASE)
DECLARATION = re.compile(r"^\s*(indicator|strategy|study|library)\s*\(")
INVISIBLE = {"\u00a0": " ", "\u2007": " ", "\u202f": " ", "\u200b": "", "\u200c": "", "\u200d": "", "\ufeff": "",
             "\u2028": "\n", "\u2029": "\n"}


def extract_pine(text):
    """Pine-koden i tekst limt inn fra en TradingView-side.

    Fjerner overskrift («Script Name», «Description», «Copy code»), linjenummer-kolonnen 1, 2, … N og
    «Expand (N lines)», og bytter usynlige web-mellomrom med vanlige. Returnerer (kode, merknad);
    merknaden er tom når teksten allerede var ren Pine-kode.
    """
    original = text
    for bad, good in INVISIBLE.items():
        text = text.replace(bad, good)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    looks_like_code = lambda part: any(VERSION.search(l) or DECLARATION.match(l) for l in part)
    # 1) Linjenummer på egne linjer (1, 2, 3 … N) etterfulgt av de N kodelinjene.
    best, k = None, 0
    while k < len(lines):
        if lines[k].strip() == "1":
            j, n = k, 0
            while j < len(lines) and lines[j].strip() == str(n + 1):
                n, j = n + 1, j + 1
            if n >= 3 and looks_like_code(lines[j:j + n]) and (best is None or n > best[1]):
                best = (j, n)
            k = max(j, k + 1)
        else:
            k += 1
    if best is not None:
        start, n = best
        code = lines[start:start + n]
        cut = next((i for i, line in enumerate(code) if EXPAND.match(line)), None)
        code = code[:cut] if cut is not None else code
        return "\n".join(code).rstrip() + "\n", (
            f"Fant Pine-koden i teksten ({len(code)} linjer). Overskrift, «Copy code» og linjenummer er fjernet.")
    # 2) Ellers: koden starter med kommentarene rett over //@version og slutter før «Expand (N lines)».
    version = next((i for i, line in enumerate(lines) if VERSION.search(line)), None)
    if version is None:
        return text, ("Usynlige tegn fra nettsiden er byttet ut." if text != original else "")
    start = version
    while start > 0 and (lines[start - 1].strip().startswith("//") or not lines[start - 1].strip()):
        start -= 1
    end = next((i for i in range(version, len(lines)) if EXPAND.match(lines[i])), len(lines))
    if start == 0 and end == len(lines):
        return text, ("Usynlige tegn fra nettsiden er byttet ut." if text != original else "")
    code = lines[start:end]
    return "\n".join(code).strip("\n") + "\n", "Fant Pine-koden i teksten; tekst før og etter koden er fjernet."


def parse(source):
    """Pine-kildekode → ``Program``. Versjonen leses fra ``//@version=N`` (mangler den: v5)."""
    if len(source) > 500_000:
        raise PineError("skriptet er for langt (maks 500 000 tegn)")
    match = VERSION.search(source)
    version = int(match.group(1)) if match else 5
    if version not in (3, 4, 5, 6):
        raise PineError(f"Pine-versjon {version} støttes ikke (bruk v4, v5 eller v6)")
    lines = logical_lines(source)
    if not lines:
        raise PineError("skriptet er tomt")
    parser = Parser(lines)
    body = parser.block(0)
    return Program(version, body, len(source.splitlines()))
