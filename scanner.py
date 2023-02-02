import string
from enum import Enum
from typing import List, Optional, Set, Dict, Tuple


class State:
    """
    A class used to represent a state in scanner's DFA.
    """

    def __init__(self, number: int):
        """ Inits State.

        :arg number: the number of the state
        """
        self._number: int = number
        self._out_transitions: List[Transition] = []
        self._is_terminal: bool = False
        self._is_lookahead: bool = False

    @property
    def number(self) -> int:
        """Return the number of the state."""
        return self._number

    @property
    def out_transitions(self) -> List["Transition"]:
        """Return transitions sourcing from the state."""
        return self._out_transitions

    @property
    def is_terminal(self) -> bool:
        """Determines if the state is terminal or not."""
        return self._is_terminal

    @property
    def is_lookahead(self) -> bool:
        """Determines if the state is lookahead or not."""
        return self._is_lookahead

    def add_transition(self, transition: "Transition"):
        """Adds a transition to the state.

        :arg transition: transition to be added"""
        self._out_transitions.append(transition)

    def set_terminal(self):
        """Sets the state as terminal."""
        self._is_terminal = True

    def set_lookahead(self):
        """Sets the state as lookahead."""
        self._is_lookahead = True


class Transition:
    """
    A class used to represent a transition in scanner's DFA.
    """

    def __init__(self, source: State, dest: State, charset: set):
        """Inits Transition.

        :arg source: the source state
        :arg dest: the destination state
        :arg charset: the characters which make the transition happen
        """
        self._source: State = source
        self._dest: State = dest
        self._charset: Set[str] = charset

    @property
    def source(self) -> State:
        """Return the source state"""
        return self._source

    @property
    def dest(self) -> State:
        """Return the destination state"""
        return self._dest

    @property
    def charset(self) -> Set[str]:
        """Return the characters which make the transition happen"""
        return self._charset


class ErrorType(Enum):
    NO_TRANSITION = 1
    INCOMPLETE_TOKEN = 2


class Error:
    """
    A class used to represent an error in scanner.
    """

    def __init__(self, title: str, content: str, line_number: int):
        """Inits Error.

        :arg title: str: the title of the error
        :arg content: str: the content of the error
        :arg line_number: int: the line number of the error
        """
        self._title: str = title
        self._content: str = content
        self._line_number: int = line_number

    @property
    def title(self) -> str:
        """Return the title of the error"""
        return self._title

    @property
    def content(self) -> str:
        """Return the content of the error"""
        return self._content

    @property
    def line_number(self) -> int:
        """Return the line number of the error"""
        return self._line_number


class Scanner:
    """
    A C-Minus compiler's scanner.

    Attributes:
        NUM, ID, KEYWORD, SYMBOL, COMMENT, WHITESPACE, EOF   token types
        EOF_symbol  the symbol used for EOF
    """
    # token types
    NUM: str = "NUM"
    ID: str = "ID"
    KEYWORD: str = "KEYWORD"
    SYMBOL: str = "SYMBOL"
    COMMENT: str = "COMMENT"
    WHITESPACE: str = "WHITESPACE"
    EOF: str = "EOF"

    EOF_symbol: str = "$"

    # character sets
    _EOF_char = None
    _all_chars: Set[str] = set(chr(i) for i in range(128))
    _digits: Set[str] = set(string.digits)
    _letters: Set[str] = set(string.ascii_letters)
    _alphanumerics: Set[str] = _digits.union(_letters)
    _symbols: Set[str] = {';', ':', ',', '[', ']', '(', ')', '{', '}', '+', '-', '*', '/', '=', '<'}
    _whitespaces: Set[str] = {' ', '\n', '\r', '\t', '\v', '\f'}
    _valid_chars: Set[str] = _alphanumerics.union(_symbols, _whitespaces)

    _keywords = {"if", "else", "void", "int", "while", "break", "switch", "default", "case", "return", "endif"}

    def __init__(self, buffer_size=1024):
        """Inits Scanner

        :arg buffer_size: size of the input buffer
        """
        # symbol table
        self.scope_stack = [0]
        self.symbol_table: Dict[str, list] = {"lexeme": [], "type": [], "size": [], "data_type": [], "scope": [], "address": []}
        for keyword in Scanner._keywords:
            self.add_symbol(keyword, "keyword", 0, None, 1, None)

        # states
        self._initialize_states()
        self._current_state: State = self._states[0]

        # input
        self._input_file = open("input.txt", mode="r")
        self._buffer_size = buffer_size
        self._buffer: List[Optional[str]] = []
        self._token_buffer: List[str] = []

        # position in buffer
        self._forward: int = 0
        self._current_char: Optional[str] = None
        self._is_file_ended: bool = False
        self._line_number: int = 1

        self._errors_dict: Dict[int, List[Error]] = {}

    @property
    def _current_token(self) -> Optional[str]:
        if self._token_buffer is not None:
            return ''.join(self._token_buffer)
        else:
            return None

    @property
    def line_number(self) -> int:
        """Return current line number."""
        return self._line_number

    def _initialize_states(self):
        """Creates states and transitions"""
        # create states
        self._states: List[State] = [State(i) for i in range(19)]

        # set terminal states
        for i in (2, 4, 5, 7, 8, 10, 12, 15, 16, 18):
            self._states[i].set_terminal()

        # set lookahead states
        for i in (2, 4, 8, 10, 12, 18):
            self._states[i].set_lookahead()

        # add transitions
        self._states[0].add_transition(
            Transition(self._states[0], self._states[1], self._digits)
        )
        self._states[0].add_transition(
            Transition(self._states[0], self._states[3], self._letters)
        )
        self._states[0].add_transition(
            Transition(self._states[0], self._states[5], self._symbols - {'/', '=', '*'})
        )
        self._states[0].add_transition(
            Transition(self._states[0], self._states[6], {'='})
        )
        self._states[0].add_transition(
            Transition(self._states[0], self._states[9], {'/'})
        )
        self._states[0].add_transition(
            Transition(self._states[0], self._states[16], self._whitespaces)
        )
        self._states[0].add_transition(
            Transition(self._states[0], self._states[17], {'*'})
        )
        self._states[1].add_transition(
            Transition(self._states[1], self._states[1], self._digits)
        )
        self._states[1].add_transition(
            Transition(self._states[1], self._states[2], self._symbols.union(self._whitespaces, {self._EOF_char}))
        )
        self._states[3].add_transition(
            Transition(self._states[3], self._states[3], self._alphanumerics)
        )
        self._states[3].add_transition(
            Transition(self._states[3], self._states[4], self._symbols.union(self._whitespaces, {self._EOF_char}))
        )
        self._states[6].add_transition(
            Transition(self._states[6], self._states[7], {'='})
        )
        self._states[6].add_transition(
            Transition(self._states[6], self._states[8], self._valid_chars.union({self._EOF_char}) - {'='})
        )
        self._states[9].add_transition(
            Transition(self._states[9], self._states[10], self._valid_chars.union({self._EOF_char}) - {'*', '/'})
        )
        self._states[9].add_transition(
            Transition(self._states[9], self._states[11], {'/'})
        )
        self._states[9].add_transition(
            Transition(self._states[9], self._states[13], {'*'})
        )
        self._states[11].add_transition(
            Transition(self._states[11], self._states[11], self._all_chars - {'\n'})
        )
        self._states[11].add_transition(
            Transition(self._states[11], self._states[12], {'\n', self._EOF_char})
        )
        self._states[13].add_transition(
            Transition(self._states[13], self._states[13], self._all_chars - {'*'})
        )
        self._states[13].add_transition(
            Transition(self._states[13], self._states[14], {'*'})
        )
        self._states[14].add_transition(
            Transition(self._states[14], self._states[15], {'/'})
        )
        self._states[14].add_transition(
            Transition(self._states[14], self._states[13], self._all_chars - {'/'})
        )
        self._states[17].add_transition(
            Transition(self._states[17], self._states[18], self._all_chars - {'/'})
        )

    def _load_buffer(self) -> List[Optional[str]]:
        """Return a list of characters of length BUFFER_SIZE.

        The characters are read from input_file.
        """
        temp = self._input_file.read(self._buffer_size)
        return [temp[i] if i < len(temp) else None for i in range(self._buffer_size)]

    def _get_next_char(self) -> str:
        """Return next character of input_file. None if EOF."""
        # check if buffer needs to be reloaded
        if self._forward == 0:
            self._buffer[:self._buffer_size] = self._load_buffer()
        elif self._forward == self._buffer_size:
            self._buffer[self._buffer_size:] = self._load_buffer()
        char = self._buffer[self._forward]

        # update forward
        self._forward += 1
        if self._forward == 2 * self._buffer_size:
            self._forward = 0

        return char

    def _decrement_forward(self):
        """Move forward back."""
        if self._forward == 0:
            self._forward = 2 * self._buffer_size - 1
        else:
            self._forward -= 1

    def _handle_error(self, error_type: ErrorType):
        """Adds occurred error to error dict."""
        error = None
        if error_type == ErrorType.NO_TRANSITION:
            if self._current_state.number == 1 and self._current_char in self._alphanumerics:
                error = Error("Invalid number", self._current_token, self._line_number)
            elif self._current_state.number == 17 and self._current_char == '/':
                error = Error("Unmatched comment", self._current_token, self._line_number)
            else:
                error = Error("Invalid input", self._current_token, self._line_number)
        elif error_type == ErrorType.INCOMPLETE_TOKEN:
            if self._current_state.number in {13, 14}:
                line_number: int = self._line_number - self._token_buffer.count('\n')
                error = Error("Unclosed comment", f"{''.join(self._token_buffer[:7])}...", line_number)

        if error is None:
            error = Error("Undefined Error!", self._current_token, self._line_number)

        if error.line_number in self._errors_dict:
            self._errors_dict[error.line_number].append(error)
        else:
            self._errors_dict[error.line_number] = [error]

    def _get_token_tuple(self) -> Tuple[str, str]:
        """Return tuple with form (token_type, token_lexeme)"""
        if self._current_state.number == 2:
            return self.NUM, self._current_token
        elif self._current_state.number == 4:
            token_type = self._get_token_type()
            return token_type, self._install_id()
        elif self._current_state.number in {5, 7, 8, 10, 18}:
            return self.SYMBOL, self._current_token
        elif self._current_state.number in {12, 15}:
            return self.COMMENT, self._current_token
        elif self._current_state.number == 16:
            return self.WHITESPACE, self._current_token

    def _get_token_type(self) -> str:
        """Return \"KEYWORD\" if current token is a keyword else \"ID\"."""
        if self._current_token in self._keywords:
            return self.KEYWORD
        return self.ID

    def _install_id(self):
        """Adds current id to symbol table if it is not."""
        token: str = self._current_token
        if token not in self.symbol_table["lexeme"]:
            self.add_symbol(token, None, None, None, None)
        return token

    def get_next_token(self) -> Tuple[str, str]:
        """Return next token of input_file."""
        if self._is_file_ended:
            return self.EOF, self.EOF_symbol

        self._token_buffer.clear()
        self._current_state = self._states[0]
        while True:
            # Check terminal
            if self._current_state.is_terminal:
                # Terminal state
                if self._current_state.is_lookahead:
                    self._token_buffer.pop()
                    self._decrement_forward()
                    # update line number
                    if self._current_char == '\n':
                        self._line_number -= 1
                token = self._get_token_tuple()
                if token[0] in {self.WHITESPACE, self.COMMENT}:
                    self._token_buffer.clear()
                    self._current_state = self._states[0]
                else:
                    return token

            # Non-terminal state
            self._current_char = self._get_next_char()
            # update line number
            if self._current_char == '\n':
                self._line_number += 1
            # file ended if EOF
            self._is_file_ended = self._current_char == self._EOF_char
            # EOF if file ended at state#0
            if self._is_file_ended and self._current_state.number == 0:
                self._input_file.close()
                return self.EOF, self.EOF_symbol
            self._token_buffer.append(self._current_char)
            # Choosing matching transition
            for transition in self._current_state.out_transitions:
                if self._current_char in transition.charset:
                    self._current_state = transition.dest
                    break
            else:  # No matching transition found -> Error
                if self._is_file_ended:
                    self._handle_error(ErrorType.INCOMPLETE_TOKEN)
                    self._input_file.close()
                    return self.EOF, self.EOF_symbol
                else:
                    self._handle_error(ErrorType.NO_TRANSITION)
                    self._token_buffer.clear()
                    self._current_state = self._states[0]

    def add_symbol(self,
                   lexeme: str,
                   symbol_type: str = None,
                   size: int = 0,
                   data_type: str = None,
                   scope: int = None,
                   address: int = None):
        """Adds a new row to the symbol table"""
        self.symbol_table["lexeme"].append(lexeme)
        self.symbol_table["type"].append(symbol_type)
        self.symbol_table["size"].append(size)
        self.symbol_table["data_type"].append(data_type)
        self.symbol_table["scope"].append(scope)
        self.symbol_table["address"].append(address)

    def update_symbol(self,
                      index: int,
                      symbol_type: str = None,
                      size: int = None,
                      data_type: str = None,
                      scope: int = None,
                      address: int = None):
        if symbol_type is not None:
            self.symbol_table["type"][index] = symbol_type
        if size is not None:
            self.symbol_table["size"][index] = size
        if data_type is not None:
            self.symbol_table["data_type"][index] = data_type
        if scope is not None:
            self.symbol_table["scope"][index] = scope
        if address is not None:
            self.symbol_table["address"][index] = address

    def get_symbol_index(self, lexeme: str) -> int:
        """Return index of the lexeme in the symbol table"""
        return self.symbol_table["lexeme"].index(lexeme)

    def pop_scope(self, scope_start: int):
        for column in self.symbol_table.values():
            column.pop(len(column) - scope_start)

    def close_input_file(self):
        self._input_file.close()

    def save_errors(self):
        """Writes errors in lexical_errors.txt."""
        with open("lexical_errors.txt", "w") as error_file:
            if len(self._errors_dict) == 0:
                error_file.write("There is no lexical error.")
            else:
                for line_num in sorted(self._errors_dict.keys()):
                    line = ''.join([f"({error.content}, {error.title}) " for error in self._errors_dict[line_num]])
                    error_file.write(f"{line_num}.\t{line}\n")

    def save_symbols(self):
        """Writes symbol table in symbol_table.txt."""
        with open("symbol_table.txt", mode="w") as symbol_table_file:
            for key, value in self.symbol_table.items():
                symbol_table_file.write(f"{value}.\t{key}\n")
