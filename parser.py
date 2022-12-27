import json
from typing import Set, Dict


class Parser:
    """
    A C-Minus compiler's parser.
    """
    def __init__(self):
        """Inits Parser"""
        self._read_table()

    def _read_table(self):
        """Reads table.json and initializes terminals, non_terminals, first_sets, follow_sets and parse_table"""
        with open("table.json", mode="r") as table_file:
            table = json.load(table_file)

        self._terminals: Set[str] = set(table["terminals"])
        self._non_terminals: Set[str] = set(table["non_terminals"])
        self._first_sets: Dict[str, Set[str]] = dict(zip(table["first"].keys(), map(set, table["first"].values())))
        self._follow_sets: Dict[str, Set[str]] = dict(zip(table["follow"].keys(), map(set, table["follow"].values())))
        self._parse_table: Dict[str, Dict[str, str]] = table["parse_table"]
