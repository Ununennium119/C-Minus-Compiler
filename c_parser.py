import json
from enum import Enum
from typing import Set, Dict, Tuple, List, Union

from anytree import Node, RenderTree

from scanner import Scanner


class ErrorType(Enum):
    ILLEGAL_TOKEN = 1
    TOKEN_DISCARDED = 2
    STACK_CORRECTION = 3
    MISSING_NON_TERMINAL = 4
    UNEXPECTED_EOF = 5


class Error:
    """
    A class used to represent an error in parser.
    """

    def __init__(self, error_type: ErrorType, subject: Union[str, Node], line_number: int):
        """Inits Error.

        :arg error_type: ErrorType: the type of the error
        :arg subject: Union[str, Tuple[str, str]]: the subject of the error
        :arg line_number: int: the line number of the error
        """
        if isinstance(subject, Node):
            subject = subject.name

        self._type: ErrorType = error_type
        self._content: str = subject
        self._line_number: int = line_number
        self._content: str = ""
        if self._type == ErrorType.ILLEGAL_TOKEN:
            self._content = f"#{line_number} : syntax error , illegal {subject}"
        elif self._type == ErrorType.TOKEN_DISCARDED:
            self._content = f"#{line_number} : syntax error , discarded {subject} from input"
        elif self._type == ErrorType.STACK_CORRECTION:
            self._content = f"syntax error , discarded {subject} from stack"
        elif self._type == ErrorType.MISSING_NON_TERMINAL:
            self._content = f"#{line_number} : syntax error , missing {subject}"
        elif self._type == ErrorType.UNEXPECTED_EOF:
            self._content = f"#{line_number} : syntax error , Unexpected EOF"
        else:
            self._content = f"Unknown error type: {error_type}"

    @property
    def type(self) -> ErrorType:
        """Return the title of the error"""
        return self._type

    @property
    def line_number(self) -> int:
        """Return the line number of the error"""
        return self._line_number

    @property
    def content(self) -> str:
        """Return the content of the error"""
        return self._content


class Parser:
    """
    A C-Minus compiler's parser.
    """
    # Actions
    _accept: str = "accept"
    _shift: str = "shift"
    _reduce: str = "reduce"
    _goto: str = "goto"

    def __init__(self, scanner: Scanner):
        """Inits Parser

        :arg scanner: Scanner: the compiler's scanner"""
        self._scanner: Scanner = scanner
        self._stack: List[Union[str, Node]] = ["0"]
        self._errors: List[Error] = []
        self._failure: bool = False

        self._update_current_token()
        self._read_table()

    @staticmethod
    def _get_rhs_count(production: List[str]) -> int:
        """Return count of rhs of the production."""
        if production[-1] == "epsilon":
            return 0
        else:
            return len(production) - 2

    def _read_table(self):
        """Initializes terminals, non_terminals, first_sets, follow_sets, grammar and parse_table."""
        with open("table.json", mode="r") as table_file:
            table: dict = json.load(table_file)

        # set of grammars terminals
        self._terminals: Set[str] = set(table["terminals"])
        # set of grammars non-terminals
        self._non_terminals: Set[str] = set(table["non_terminals"])
        # first and follow sets of non-terminals
        self._first_sets: Dict[str, Set[str]] = dict(zip(table["first"].keys(), map(set, table["first"].values())))
        self._follow_sets: Dict[str, Set[str]] = dict(zip(table["follow"].keys(), map(set, table["follow"].values())))
        # grammar's productions
        self._grammar: Dict[str, List[str]] = table["grammar"]
        # SLR parse table
        self._parse_table: Dict[str, Dict[str, Tuple[str, str]]] = dict(
            zip(table["parse_table"].keys(), map(lambda row: dict(
                zip(row.keys(), map(lambda entry: tuple(entry.split("_")), row.values()))
            ), table["parse_table"].values()))
        )

    def _update_current_token(self):
        """Stores next token in _current_token and updates _current_input."""
        self._current_token: Tuple[str, str] = self._scanner.get_next_token()
        self._current_input: str = ""
        if self._current_token[0] in {Scanner.KEYWORD, Scanner.SYMBOL, Scanner.EOF}:
            self._current_input = self._current_token[1]
        else:
            self._current_input = self._current_token[0]

    def run(self):
        """Parses the input. Return True if UNEXPECTED_EOF"""
        while True:
            # get action from parse_table
            last_state = self._stack[-1]
            try:
                action = self._parse_table[last_state].get(self._current_input)
            except KeyError:
                # invalid state
                raise Exception(f"State \"{last_state}\" does not exist.")
            if action is not None:
                # perform the action
                if action[0] == self._accept:
                    # accept
                    break
                elif action[0] == self._shift:
                    # push current_token and shift_state into the stack
                    shift_state = action[1]
                    self._stack.append(Node(f"({self._current_token[0]}, {self._current_token[1]})"))
                    self._stack.append(shift_state)

                    # get next token
                    self._update_current_token()
                elif action[0] == self._reduce:
                    # pop rhs of the production from the stack and update parse tree
                    production_number = action[1]
                    production = self._grammar[production_number]
                    production_lhs = production[0]
                    production_rhs_count = self._get_rhs_count(production)
                    production_lhs_node: Node = Node(production_lhs)
                    if production_rhs_count == 0:
                        node = Node("epsilon")
                        node.parent = production_lhs_node
                    else:
                        popped_nodes = []
                        for _ in range(production_rhs_count):
                            self._stack.pop()
                            popped_nodes.append(self._stack.pop())
                        for node in popped_nodes[::-1]:
                            node.parent = production_lhs_node

                    # push lhs of the production and goto_state into the stack
                    last_state = self._stack[-1]
                    try:
                        goto_state = self._parse_table[last_state][production_lhs][1]
                    except KeyError:
                        # problem in parse_table
                        raise Exception(f"Goto[{last_state}, {production_lhs}] is empty.")
                    self._stack.append(production_lhs_node)
                    self._stack.append(goto_state)
                else:
                    # problem in parse_table
                    raise Exception(f"Unknown action: {action}.")
            else:
                if self.handle_error():
                    # failure if UNEXPECTED_EOF
                    self._failure = True
                    break

    def handle_error(self) -> bool:
        """Handles syntax errors. Return True if error is UNEXPECTED_EOF"""
        # discard the first input
        self._errors.append(Error(ErrorType.ILLEGAL_TOKEN, self._current_token[1], self._scanner.line_number))
        self._update_current_token()

        # pop from stack until state has non-empty goto cell
        while True:
            state = self._stack[-1]
            goto_and_actions_of_current_state = self._parse_table[state].values()
            # break if the current state has a goto cell
            if any(map(lambda table_cell: table_cell[0] == self._goto,
                       goto_and_actions_of_current_state)):
                break
            discarded_state, discarded_node = self._stack.pop(), self._stack.pop()
            self._errors.append(Error(ErrorType.STACK_CORRECTION, discarded_node, self._scanner.line_number))

        goto_keys = self._get_goto_non_terminals(state)
        # discard input, while input not in any follow(non_terminal)
        selected_non_terminal = None
        while True:
            for non_terminal in goto_keys:
                if self._current_input in self._follow_sets[non_terminal]:
                    selected_non_terminal = non_terminal
                    break
            if selected_non_terminal is None:
                if self._current_input == Scanner.EOF_symbol:
                    # input is EOF, halt parser
                    self._errors.append(Error(ErrorType.UNEXPECTED_EOF, "", self._scanner.line_number))
                    return True
                else:
                    # discard input
                    self._errors.append(Error(ErrorType.TOKEN_DISCARDED, self._current_token[1], self._scanner.line_number))
                    self._update_current_token()
            else:
                # input is in follow(non_terminal)
                break
        self._stack.append(Node(selected_non_terminal))
        self._stack.append(self._parse_table[state][selected_non_terminal][1])
        self._errors.append(Error(ErrorType.MISSING_NON_TERMINAL, selected_non_terminal, self._scanner.line_number))
        return False

    def _get_goto_non_terminals(self, state: str) -> List[str]:
        """Return the non-terminals which the state has a goto with them."""
        non_terminals_of_state = []
        state_goto_and_actions = self._parse_table[state]
        for non_terminal in self._non_terminals:
            if state_goto_and_actions.get(non_terminal) is not None:
                non_terminals_of_state.append(non_terminal)
        non_terminals_of_state.sort()
        return non_terminals_of_state

    def save_parse_tree(self):
        """Writes parse tree in parse_tree.txt."""
        # empty file if failure
        if self._failure:
            with open("parse_tree.txt", mode='w') as parse_tree_file:
                parse_tree_file.write("")
                return

        root = self._stack[1]
        # add EOF node
        node = Node("$")
        node.parent = root

        # write parse tree in the file
        lines = []
        for pre, fill, node in RenderTree(root):
            lines.append(str(f"{pre}{node.name}\n"))
        with open("parse_tree.txt", mode='w', encoding="utf-8") as parse_tree_file:
            parse_tree_file.writelines(lines)

    def save_errors(self):
        """Writes errors in syntax_errors.txt."""
        with open("syntax_errors.txt", "w") as error_file:
            if len(self._errors) == 0:
                error_file.write("There is no syntax error.")
            else:
                for error in self._errors:
                    error_file.write(f"{error.content}\n")
