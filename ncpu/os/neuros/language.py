"""nCPU Simple Language (nsl) Specification.

A minimal C-like language that compiles to nCPU assembly.
Designed to be small enough for a neural compiler to learn,
but expressive enough to write real programs.

Language features:
    - Integer variables (mapped to R0-R7, max 8 per scope)
    - Arithmetic: +, -, *, /, %, &, |, ^, <<, >>
    - Comparison: ==, !=, <, >, <=, >=
    - Logical: &&, ||, !
    - Assignment: =, +=, -=, *=, /=
    - Control flow: if/else, while, for, do...while
    - Functions: fn name(args) { body }
    - Built-in: halt, print (for debugging)
    - Comments: // single-line

Grammar (EBNF):
    program     = { function | statement } ;
    function    = "fn" IDENT "(" [params] ")" block ;
    params      = IDENT { "," IDENT } ;
    block       = "{" { statement } "}" ;
    statement   = var_decl | assignment | if_stmt | while_stmt | for_stmt
                | do_while_stmt | return_stmt | call_stmt | halt_stmt | ";" ;
    var_decl    = "var" IDENT "=" expr ";" ;
    assignment  = IDENT ("=" | "+=" | "-=" | "*=" | "/=") expr ";" ;
    if_stmt     = "if" "(" expr ")" block [ "else" block ] ;
    while_stmt  = "while" "(" expr ")" block ;
    for_stmt    = "for" "(" (var_decl | assignment) expr ";" assignment_no_semi ")" block ;
    do_while    = "do" block "while" "(" expr ")" ";" ;
    return_stmt = "return" [expr] ";" ;
    call_stmt   = IDENT "(" [args] ")" ";" ;
    halt_stmt   = "halt" ";" ;
    args        = expr { "," expr } ;
    expr        = logical_or ;
    logical_or  = logical_and { "||" logical_and } ;
    logical_and = comparison { "&&" comparison } ;
    comparison  = bitwise { ("==" | "!=" | "<" | ">" | "<=" | ">=") bitwise } ;
    bitwise     = shift { ("&" | "|" | "^") shift } ;
    shift       = additive { ("<<" | ">>") additive } ;
    additive    = term { ("+" | "-") term } ;
    term        = unary { ("*" | "/" | "%") unary } ;
    unary       = ["-" | "!"] primary ;
    primary     = NUMBER | IDENT | "(" expr ")" | IDENT "(" [args] ")" ;
"""

from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum, auto


# ═══════════════════════════════════════════════════════════════════════════════
# Token Types
# ═══════════════════════════════════════════════════════════════════════════════

class TokenType(Enum):
    # Literals
    NUMBER = auto()
    IDENT = auto()

    # Keywords
    VAR = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    FOR = auto()
    DO = auto()
    FN = auto()
    RETURN = auto()
    HALT = auto()

    # Operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    AMP = auto()
    PIPE = auto()
    CARET = auto()
    SHL = auto()
    SHR = auto()
    ASSIGN = auto()
    PLUS_ASSIGN = auto()
    MINUS_ASSIGN = auto()
    STAR_ASSIGN = auto()
    SLASH_ASSIGN = auto()
    EQ = auto()
    NEQ = auto()
    LT = auto()
    GT = auto()
    LTE = auto()
    GTE = auto()
    LAND = auto()
    LOR = auto()
    BANG = auto()

    # Punctuation
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    COMMA = auto()
    SEMI = auto()

    # Special
    EOF = auto()


KEYWORDS = {
    "var": TokenType.VAR,
    "if": TokenType.IF,
    "else": TokenType.ELSE,
    "while": TokenType.WHILE,
    "for": TokenType.FOR,
    "do": TokenType.DO,
    "fn": TokenType.FN,
    "return": TokenType.RETURN,
    "halt": TokenType.HALT,
}


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    col: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# AST Nodes
# ═══════════════════════════════════════════════════════════════════════════════

class ASTNode:
    """Base AST node."""
    pass


@dataclass
class NumberLit(ASTNode):
    value: int
    line: int = 0


@dataclass
class IdentExpr(ASTNode):
    name: str
    line: int = 0


@dataclass
class UnaryExpr(ASTNode):
    op: str        # "-"
    operand: ASTNode
    line: int = 0


@dataclass
class BinaryExpr(ASTNode):
    op: str        # +, -, *, /, &, |, ^, <<, >>, ==, !=, <, >, <=, >=
    left: ASTNode
    right: ASTNode
    line: int = 0


@dataclass
class CallExpr(ASTNode):
    name: str
    args: List[ASTNode] = field(default_factory=list)
    line: int = 0


@dataclass
class VarDecl(ASTNode):
    name: str
    init: ASTNode
    line: int = 0


@dataclass
class Assignment(ASTNode):
    name: str
    value: ASTNode
    line: int = 0


@dataclass
class IfStmt(ASTNode):
    condition: ASTNode
    then_block: List[ASTNode]
    else_block: Optional[List[ASTNode]] = None
    line: int = 0


@dataclass
class WhileStmt(ASTNode):
    condition: ASTNode
    body: List[ASTNode]
    line: int = 0


@dataclass
class ForStmt(ASTNode):
    init: ASTNode              # VarDecl or Assignment
    condition: ASTNode         # loop condition expression
    update: ASTNode            # Assignment (no trailing semicolon)
    body: List[ASTNode]
    line: int = 0


@dataclass
class DoWhileStmt(ASTNode):
    body: List[ASTNode]
    condition: ASTNode
    line: int = 0


@dataclass
class ReturnStmt(ASTNode):
    value: Optional[ASTNode] = None
    line: int = 0


@dataclass
class HaltStmt(ASTNode):
    line: int = 0


@dataclass
class FuncDecl(ASTNode):
    name: str
    params: List[str]
    body: List[ASTNode]
    line: int = 0


@dataclass
class Program(ASTNode):
    """Top-level program node."""
    functions: List[FuncDecl] = field(default_factory=list)
    statements: List[ASTNode] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# Lexer
# ═══════════════════════════════════════════════════════════════════════════════

class Lexer:
    """Tokenize nsl source code."""

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens: List[Token] = []

    def tokenize(self) -> List[Token]:
        """Tokenize the entire source."""
        while self.pos < len(self.source):
            self._skip_whitespace_and_comments()
            if self.pos >= len(self.source):
                break

            ch = self.source[self.pos]

            # Numbers
            if ch.isdigit():
                self._read_number()
            # Identifiers and keywords
            elif ch.isalpha() or ch == '_':
                self._read_ident()
            # Two-character operators
            elif ch == '=' and self._peek(1) == '=':
                self.tokens.append(Token(TokenType.EQ, "==", self.line, self.col))
                self._advance(2)
            elif ch == '!' and self._peek(1) == '=':
                self.tokens.append(Token(TokenType.NEQ, "!=", self.line, self.col))
                self._advance(2)
            elif ch == '<' and self._peek(1) == '=':
                self.tokens.append(Token(TokenType.LTE, "<=", self.line, self.col))
                self._advance(2)
            elif ch == '>' and self._peek(1) == '=':
                self.tokens.append(Token(TokenType.GTE, ">=", self.line, self.col))
                self._advance(2)
            elif ch == '<' and self._peek(1) == '<':
                self.tokens.append(Token(TokenType.SHL, "<<", self.line, self.col))
                self._advance(2)
            elif ch == '>' and self._peek(1) == '>':
                self.tokens.append(Token(TokenType.SHR, ">>", self.line, self.col))
                self._advance(2)
            elif ch == '&' and self._peek(1) == '&':
                self.tokens.append(Token(TokenType.LAND, "&&", self.line, self.col))
                self._advance(2)
            elif ch == '|' and self._peek(1) == '|':
                self.tokens.append(Token(TokenType.LOR, "||", self.line, self.col))
                self._advance(2)
            elif ch == '+' and self._peek(1) == '=':
                self.tokens.append(Token(TokenType.PLUS_ASSIGN, "+=", self.line, self.col))
                self._advance(2)
            elif ch == '-' and self._peek(1) == '=':
                self.tokens.append(Token(TokenType.MINUS_ASSIGN, "-=", self.line, self.col))
                self._advance(2)
            elif ch == '*' and self._peek(1) == '=':
                self.tokens.append(Token(TokenType.STAR_ASSIGN, "*=", self.line, self.col))
                self._advance(2)
            elif ch == '/' and self._peek(1) == '=':
                self.tokens.append(Token(TokenType.SLASH_ASSIGN, "/=", self.line, self.col))
                self._advance(2)
            # Single-character operators
            elif ch == '=':
                self.tokens.append(Token(TokenType.ASSIGN, "=", self.line, self.col))
                self._advance(1)
            elif ch == '+':
                self.tokens.append(Token(TokenType.PLUS, "+", self.line, self.col))
                self._advance(1)
            elif ch == '-':
                self.tokens.append(Token(TokenType.MINUS, "-", self.line, self.col))
                self._advance(1)
            elif ch == '*':
                self.tokens.append(Token(TokenType.STAR, "*", self.line, self.col))
                self._advance(1)
            elif ch == '/':
                self.tokens.append(Token(TokenType.SLASH, "/", self.line, self.col))
                self._advance(1)
            elif ch == '%':
                self.tokens.append(Token(TokenType.PERCENT, "%", self.line, self.col))
                self._advance(1)
            elif ch == '&':
                self.tokens.append(Token(TokenType.AMP, "&", self.line, self.col))
                self._advance(1)
            elif ch == '|':
                self.tokens.append(Token(TokenType.PIPE, "|", self.line, self.col))
                self._advance(1)
            elif ch == '!':
                self.tokens.append(Token(TokenType.BANG, "!", self.line, self.col))
                self._advance(1)
            elif ch == '^':
                self.tokens.append(Token(TokenType.CARET, "^", self.line, self.col))
                self._advance(1)
            elif ch == '<':
                self.tokens.append(Token(TokenType.LT, "<", self.line, self.col))
                self._advance(1)
            elif ch == '>':
                self.tokens.append(Token(TokenType.GT, ">", self.line, self.col))
                self._advance(1)
            # Punctuation
            elif ch == '(':
                self.tokens.append(Token(TokenType.LPAREN, "(", self.line, self.col))
                self._advance(1)
            elif ch == ')':
                self.tokens.append(Token(TokenType.RPAREN, ")", self.line, self.col))
                self._advance(1)
            elif ch == '{':
                self.tokens.append(Token(TokenType.LBRACE, "{", self.line, self.col))
                self._advance(1)
            elif ch == '}':
                self.tokens.append(Token(TokenType.RBRACE, "}", self.line, self.col))
                self._advance(1)
            elif ch == ',':
                self.tokens.append(Token(TokenType.COMMA, ",", self.line, self.col))
                self._advance(1)
            elif ch == ';':
                self.tokens.append(Token(TokenType.SEMI, ";", self.line, self.col))
                self._advance(1)
            else:
                raise SyntaxError(f"Unexpected character '{ch}' at line {self.line}")

        self.tokens.append(Token(TokenType.EOF, "", self.line, self.col))
        return self.tokens

    def _peek(self, offset: int = 0) -> str:
        pos = self.pos + offset
        if pos < len(self.source):
            return self.source[pos]
        return '\0'

    def _advance(self, count: int = 1):
        for _ in range(count):
            if self.pos < len(self.source):
                if self.source[self.pos] == '\n':
                    self.line += 1
                    self.col = 1
                else:
                    self.col += 1
                self.pos += 1

    def _skip_whitespace_and_comments(self):
        while self.pos < len(self.source):
            ch = self.source[self.pos]
            if ch in (' ', '\t', '\r', '\n'):
                self._advance()
            elif ch == '/' and self._peek(1) == '/':
                while self.pos < len(self.source) and self.source[self.pos] != '\n':
                    self._advance()
            else:
                break

    def _read_number(self):
        start = self.pos
        col = self.col
        # Hex
        if self.source[self.pos] == '0' and self.pos + 1 < len(self.source) and self.source[self.pos + 1] in ('x', 'X'):
            self._advance(2)
            while self.pos < len(self.source) and self.source[self.pos] in '0123456789abcdefABCDEF':
                self._advance()
        else:
            while self.pos < len(self.source) and self.source[self.pos].isdigit():
                self._advance()
        text = self.source[start:self.pos]
        self.tokens.append(Token(TokenType.NUMBER, text, self.line, col))

    def _read_ident(self):
        start = self.pos
        col = self.col
        while self.pos < len(self.source) and (self.source[self.pos].isalnum() or self.source[self.pos] == '_'):
            self._advance()
        text = self.source[start:self.pos]
        tt = KEYWORDS.get(text, TokenType.IDENT)
        self.tokens.append(Token(tt, text, self.line, col))


# ═══════════════════════════════════════════════════════════════════════════════
# Parser (Recursive Descent)
# ═══════════════════════════════════════════════════════════════════════════════

class Parser:
    """Recursive descent parser for nsl."""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> Program:
        """Parse tokens into a Program AST."""
        program = Program()
        while not self._at_end():
            if self._check(TokenType.FN):
                program.functions.append(self._parse_function())
            else:
                stmt = self._parse_statement()
                if stmt is not None:
                    program.statements.append(stmt)
        return program

    def _current(self) -> Token:
        return self.tokens[self.pos]

    def _at_end(self) -> bool:
        return self._current().type == TokenType.EOF

    def _check(self, tt: TokenType) -> bool:
        return self._current().type == tt

    def _advance(self) -> Token:
        tok = self._current()
        if not self._at_end():
            self.pos += 1
        return tok

    def _expect(self, tt: TokenType, msg: str = "") -> Token:
        if self._current().type != tt:
            tok = self._current()
            raise SyntaxError(
                f"Expected {tt.name} but got {tok.type.name} ('{tok.value}') "
                f"at line {tok.line}" + (f": {msg}" if msg else ""))
        return self._advance()

    def _match(self, *types: TokenType) -> Optional[Token]:
        if self._current().type in types:
            return self._advance()
        return None

    # ─── Declarations ──────────────────────────────────────────────────

    def _parse_function(self) -> FuncDecl:
        self._expect(TokenType.FN)
        name = self._expect(TokenType.IDENT).value
        self._expect(TokenType.LPAREN)
        params = []
        if not self._check(TokenType.RPAREN):
            params.append(self._expect(TokenType.IDENT).value)
            while self._match(TokenType.COMMA):
                params.append(self._expect(TokenType.IDENT).value)
        self._expect(TokenType.RPAREN)
        body = self._parse_block()
        return FuncDecl(name=name, params=params, body=body)

    def _parse_block(self) -> List[ASTNode]:
        self._expect(TokenType.LBRACE)
        stmts = []
        while not self._check(TokenType.RBRACE) and not self._at_end():
            stmt = self._parse_statement()
            if stmt is not None:
                stmts.append(stmt)
        self._expect(TokenType.RBRACE)
        return stmts

    # ─── Statements ────────────────────────────────────────────────────

    def _parse_statement(self) -> Optional[ASTNode]:
        if self._match(TokenType.SEMI):
            return None

        if self._check(TokenType.VAR):
            return self._parse_var_decl()
        if self._check(TokenType.IF):
            return self._parse_if()
        if self._check(TokenType.WHILE):
            return self._parse_while()
        if self._check(TokenType.FOR):
            return self._parse_for()
        if self._check(TokenType.DO):
            return self._parse_do_while()
        if self._check(TokenType.RETURN):
            return self._parse_return()
        if self._check(TokenType.HALT):
            return self._parse_halt()

        # Assignment or expression statement (function call)
        if self._check(TokenType.IDENT):
            # Look ahead for compound assignment operators
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type in (
                    TokenType.PLUS_ASSIGN, TokenType.MINUS_ASSIGN,
                    TokenType.STAR_ASSIGN, TokenType.SLASH_ASSIGN):
                return self._parse_compound_assignment()
            # Look ahead for simple assignment
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == TokenType.ASSIGN:
                return self._parse_assignment()
            # Function call as statement
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == TokenType.LPAREN:
                expr = self._parse_expr()
                self._expect(TokenType.SEMI)
                return expr
            # Otherwise it must be an assignment
            return self._parse_assignment()

        raise SyntaxError(
            f"Unexpected token {self._current().type.name} ('{self._current().value}') "
            f"at line {self._current().line}")

    def _parse_var_decl(self) -> VarDecl:
        self._expect(TokenType.VAR)
        name = self._expect(TokenType.IDENT).value
        self._expect(TokenType.ASSIGN)
        init = self._parse_expr()
        self._expect(TokenType.SEMI)
        return VarDecl(name=name, init=init)

    def _parse_assignment(self) -> Assignment:
        name = self._expect(TokenType.IDENT).value
        self._expect(TokenType.ASSIGN)
        value = self._parse_expr()
        self._expect(TokenType.SEMI)
        return Assignment(name=name, value=value)

    def _parse_if(self) -> IfStmt:
        self._expect(TokenType.IF)
        self._expect(TokenType.LPAREN)
        cond = self._parse_expr()
        self._expect(TokenType.RPAREN)
        then_block = self._parse_block()
        else_block = None
        if self._match(TokenType.ELSE):
            else_block = self._parse_block()
        return IfStmt(condition=cond, then_block=then_block, else_block=else_block)

    def _parse_while(self) -> WhileStmt:
        self._expect(TokenType.WHILE)
        self._expect(TokenType.LPAREN)
        cond = self._parse_expr()
        self._expect(TokenType.RPAREN)
        body = self._parse_block()
        return WhileStmt(condition=cond, body=body)

    def _parse_for(self) -> ForStmt:
        tok = self._expect(TokenType.FOR)
        self._expect(TokenType.LPAREN)
        # Init: var_decl or assignment (both consume their trailing semicolon)
        if self._check(TokenType.VAR):
            init = self._parse_var_decl()
        else:
            init = self._parse_assignment()
        # Condition expression followed by semicolon
        cond = self._parse_expr()
        self._expect(TokenType.SEMI)
        # Update: assignment WITHOUT trailing semicolon
        update = self._parse_assignment_no_semi()
        self._expect(TokenType.RPAREN)
        body = self._parse_block()
        return ForStmt(init=init, condition=cond, update=update, body=body, line=tok.line)

    def _parse_do_while(self) -> DoWhileStmt:
        tok = self._expect(TokenType.DO)
        body = self._parse_block()
        self._expect(TokenType.WHILE)
        self._expect(TokenType.LPAREN)
        cond = self._parse_expr()
        self._expect(TokenType.RPAREN)
        self._expect(TokenType.SEMI)
        return DoWhileStmt(body=body, condition=cond, line=tok.line)

    def _parse_assignment_no_semi(self) -> Assignment:
        """Parse an assignment without consuming a trailing semicolon.
        Used inside for-loop update clauses."""
        name = self._expect(TokenType.IDENT).value
        # Support compound assignment in for-loop update
        compound_tok = self._match(
            TokenType.PLUS_ASSIGN, TokenType.MINUS_ASSIGN,
            TokenType.STAR_ASSIGN, TokenType.SLASH_ASSIGN)
        if compound_tok:
            op_map = {"+=": "+", "-=": "-", "*=": "*", "/=": "/"}
            op = op_map[compound_tok.value]
            value = self._parse_expr()
            return Assignment(
                name=name,
                value=BinaryExpr(op=op, left=IdentExpr(name=name), right=value))
        self._expect(TokenType.ASSIGN)
        value = self._parse_expr()
        return Assignment(name=name, value=value)

    def _parse_compound_assignment(self) -> Assignment:
        """Parse compound assignment: x += expr; x -= expr; etc."""
        name = self._expect(TokenType.IDENT).value
        tok = self._advance()  # consume the compound operator
        op_map = {"+=": "+", "-=": "-", "*=": "*", "/=": "/"}
        op = op_map[tok.value]
        value = self._parse_expr()
        self._expect(TokenType.SEMI)
        # Desugar: x += expr  ->  x = x + expr
        return Assignment(
            name=name,
            value=BinaryExpr(op=op, left=IdentExpr(name=name), right=value))

    def _parse_return(self) -> ReturnStmt:
        self._expect(TokenType.RETURN)
        value = None
        if not self._check(TokenType.SEMI):
            value = self._parse_expr()
        self._expect(TokenType.SEMI)
        return ReturnStmt(value=value)

    def _parse_halt(self) -> HaltStmt:
        tok = self._expect(TokenType.HALT)
        self._expect(TokenType.SEMI)
        return HaltStmt(line=tok.line)

    # ─── Expressions (Precedence Climbing) ─────────────────────────────

    def _parse_expr(self) -> ASTNode:
        return self._parse_logical_or()

    def _parse_logical_or(self) -> ASTNode:
        left = self._parse_logical_and()
        while True:
            tok = self._match(TokenType.LOR)
            if tok is None:
                break
            right = self._parse_logical_and()
            left = BinaryExpr(op="||", left=left, right=right)
        return left

    def _parse_logical_and(self) -> ASTNode:
        left = self._parse_comparison()
        while True:
            tok = self._match(TokenType.LAND)
            if tok is None:
                break
            right = self._parse_comparison()
            left = BinaryExpr(op="&&", left=left, right=right)
        return left

    def _parse_comparison(self) -> ASTNode:
        left = self._parse_bitwise()
        while True:
            tok = self._match(TokenType.EQ, TokenType.NEQ, TokenType.LT,
                              TokenType.GT, TokenType.LTE, TokenType.GTE)
            if tok is None:
                break
            right = self._parse_bitwise()
            left = BinaryExpr(op=tok.value, left=left, right=right)
        return left

    def _parse_bitwise(self) -> ASTNode:
        left = self._parse_shift()
        while True:
            tok = self._match(TokenType.AMP, TokenType.PIPE, TokenType.CARET)
            if tok is None:
                break
            right = self._parse_shift()
            left = BinaryExpr(op=tok.value, left=left, right=right)
        return left

    def _parse_shift(self) -> ASTNode:
        left = self._parse_additive()
        while True:
            tok = self._match(TokenType.SHL, TokenType.SHR)
            if tok is None:
                break
            right = self._parse_additive()
            left = BinaryExpr(op=tok.value, left=left, right=right)
        return left

    def _parse_additive(self) -> ASTNode:
        left = self._parse_term()
        while True:
            tok = self._match(TokenType.PLUS, TokenType.MINUS)
            if tok is None:
                break
            right = self._parse_term()
            left = BinaryExpr(op=tok.value, left=left, right=right)
        return left

    def _parse_term(self) -> ASTNode:
        left = self._parse_unary()
        while True:
            tok = self._match(TokenType.STAR, TokenType.SLASH, TokenType.PERCENT)
            if tok is None:
                break
            right = self._parse_unary()
            left = BinaryExpr(op=tok.value, left=left, right=right)
        return left

    def _parse_unary(self) -> ASTNode:
        if self._match(TokenType.MINUS):
            operand = self._parse_unary()
            return UnaryExpr(op="-", operand=operand)
        if self._match(TokenType.BANG):
            operand = self._parse_unary()
            return UnaryExpr(op="!", operand=operand)
        return self._parse_primary()

    def _parse_primary(self) -> ASTNode:
        # Number literal
        tok = self._match(TokenType.NUMBER)
        if tok:
            if tok.value.startswith(('0x', '0X')):
                return NumberLit(value=int(tok.value, 16), line=tok.line)
            return NumberLit(value=int(tok.value), line=tok.line)

        # Identifier or function call
        if self._check(TokenType.IDENT):
            name_tok = self._advance()
            # Function call
            if self._match(TokenType.LPAREN):
                args = []
                if not self._check(TokenType.RPAREN):
                    args.append(self._parse_expr())
                    while self._match(TokenType.COMMA):
                        args.append(self._parse_expr())
                self._expect(TokenType.RPAREN)
                return CallExpr(name=name_tok.value, args=args, line=name_tok.line)
            return IdentExpr(name=name_tok.value, line=name_tok.line)

        # Parenthesized expression
        if self._match(TokenType.LPAREN):
            expr = self._parse_expr()
            self._expect(TokenType.RPAREN)
            return expr

        raise SyntaxError(
            f"Expected expression but got {self._current().type.name} "
            f"('{self._current().value}') at line {self._current().line}")
