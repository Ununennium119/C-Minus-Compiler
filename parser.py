import json
from typing import Set, Dict, Tuple, List

from scanner import Scanner


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
        """Inits Parser"""
        self._scanner: Scanner = scanner
        self._stack: list = ["0"]

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
        """Reads table.json and initializes terminals, non_terminals, first_sets, follow_sets and parse_table."""
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
        self._current_token = self._scanner.get_next_token()
        if self._current_token[0] in {Scanner.KEYWORD, Scanner.SYMBOL, Scanner.EOF}:
            self._current_input = self._current_token[1]
        else:
            self._current_input = self._current_token[0]

    def run(self):
        while True:
            print("--------------------------------------------------------------------------------")
            print(f"Stack: {self._stack}")
            print(f"Current Token: {self._current_token}")
            # get action from parse_table
            last_state = self._stack[-1]
            try:
                action = self._parse_table[last_state].get(self._current_input)
            except KeyError as e:
                raise Exception(f"State \"{last_state}\" does not exist.")
            if action is not None:
                print(f"Action: {action}")
                if action[0] == self._accept:
                    # accept
                    print("Parsing finished!")
                    break
                elif action[0] == self._shift:
                    # push current_input and shift_state into the stack
                    shift_state = action[1]
                    print(f"Pushing {self._current_input} {action[1]} into the stack...")
                    self._stack.append(self._current_input)
                    self._stack.append(shift_state)

                    # get next token
                    self._update_current_token()
                elif action[0] == self._reduce:
                    # pop rhs of the production from the stack
                    production_number = action[1]
                    production = self._grammar[production_number]
                    print(f"Applying production {' '.join(production)}...")
                    print("Popping", end=" ")
                    for _ in range(self._get_rhs_count(production) * 2):
                        print(self._stack.pop(), end=" ")
                    print("from stack...")

                    # push lhs of the production and goto_state into the stack
                    production_lhs = production[0]
                    last_state = self._stack[-1]
                    try:
                        goto_state = self._parse_table[last_state][production_lhs][1]
                    except KeyError:
                        raise Exception(f"Goto[{last_state}, {production_lhs}] is empty.")
                    print(f"Pushing {production_lhs} {goto_state} into the stack...")
                    self._stack.append(production_lhs)
                    self._stack.append(goto_state)
                else:
                    raise Exception(f"Unknown action: {action}.")
            else:
                print("Error!")
                self._scanner.close_input_file()
                break
