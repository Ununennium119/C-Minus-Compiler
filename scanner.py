import string
from enum import Enum
from typing import List, Optional, Set, Dict, Tuple


class State:
    """
    A class used to represent a state in scanner's DFA.

    Attributes:
        number : int: the number of the state
        out_transitions : List[Transition]: transitions sourcing from the state
        is_terminal : bool: determines if the state is terminal or not
        is_lookahead : bool: determines if the state is lookahead or not
    """

    def __init__(self, number: int):
        """ Inits State

        :arg number: the number of the state
        """
        self.number: int = number
        self.out_transitions: List[Transition] = []
        self.is_terminal: bool = False
        self.is_lookahead: bool = False

    def add_transition(self, transition: "Transition"):
        """Adds a transition to the state.

        :arg transition: transition to be added"""
        self.out_transitions.append(transition)

    def set_terminal(self):
        """Sets the state as terminal."""
        self.is_terminal = True

    def set_lookahead(self):
        """Sets the state as lookahead."""
        self.is_lookahead = True


class Transition:
    """
    A class used to represent a transition in scanner's DFA.

    Attributes:
        source: State: the source state
        dest: State: the destination state
        charset: Set[str]: the characters which make the transition happen
    """

    def __init__(self, source: State, dest: State, charset: set):
        """Inits Transition

        :arg source: the source state
        :arg dest: the destination state
        :arg charset: the characters which make the transition happen
        """
        self.source: State = source
        self.dest: State = dest
        self.charset: Set[str] = charset


class Error(Enum):
    NO_TRANSITION = 1
    INCOMPLETE_TOKEN = 2


class Scanner:
    BUFFER_SIZE = 1024

    EOF = None
    all_chars: Set[str] = set(chr(i) for i in range(128))
    digits: Set[str] = set(string.digits)
    letters: Set[str] = set(string.ascii_letters)
    alphanumerics: Set[str] = digits.union(letters)
    symbols: Set[str] = {';', ':', ',', '[', ']', '(', ')', '{', '}', '+', '-', '*', '/', '=', '<'}
    whitespaces: Set[str] = {' ', '\n', '\r', '\t', '\v', '\f'}
    valid_chars: Set[str] = alphanumerics.union(symbols, whitespaces)

    keywords = {"if", "else", "void", "int", "while", "break", "switch", "default", "case", "return", "endif"}

    def __init__(self):
        """Inits Scanner"""
        """ Input storage"""
        self.input_file = open("input.txt", mode="r")
        self.buffer: List[Optional[str]] = []

        """ Error storage"""
        self.error_file = open("lexical_errors.txt", mode="w")
        self.error_count = 0

        """ Symbol Table """
        self.symbol_table: Dict[str, List[Optional]] = {}
        self._initialize_symbol_table()
        self.symbol_table_file = open("symbol_table.txt", mode="w")

        """ Position in buffer """
        self.forward: int = 0
        self.current_char: str = None

        """ Token temporary storage """
        self.token_buffer: str = ""

        """ State"""
        self._initialize_states()
        self.current_state: State = self.states[0]

    def handle_error(self, error: Error):
        # TODO: Also implement Panic Mode
        self.error_count += 1
        if error == Error.NO_TRANSITION:
            if self.current_state == self.states[1] and self.current_char in self.alphanumerics:
                error_tuple = (self.token_buffer, "Invalid number")
            elif self.current_state == self.states[17] and self.current_char == '/':
                error_tuple = (self.token_buffer, "Unmatched comment")
            else:
                error_tuple = (self.token_buffer, "Invalid input")
        elif error == Error.INCOMPLETE_TOKEN:
            if self.current_state == self.states[13] or \
                    self.current_state == self.states[14]:
                error_tuple = (self.token_buffer[:min(9, len(self.token_buffer))] + "...", "Unclosed comment")
        else:
            error_tuple = (self.token_buffer, "Undefined Error!")
        self.error_file.write(str(error_tuple) + "\n")

    def token_tuple(self):
        if self.current_state == self.states[2]:
            return "NUM", self.token_buffer
        elif self.current_state == self.states[4]:
            token_type = self.get_token_type()
            return token_type, self.install_id(token_type)
        elif self.current_state == self.states[5] or \
                self.current_state == self.states[7] or \
                self.current_state == self.states[8] or \
                self.current_state == self.states[10] or \
                self.current_state == self.states[18]:
            return "SYMBOL", self.token_buffer
        elif self.current_state == self.states[12] or \
                self.current_state == self.states[15]:
            return "COMMENT", self.token_buffer
        elif self.current_state == self.states[16]:
            return "WHITESPACE", self.token_buffer

    def get_token_type(self):
        if self.token_buffer in Scanner.keywords:
            return "KEYWORD"
        return "ID"

    def install_id(self, token_type):
        if token_type == "ID":
            if not self.token_buffer in self.symbol_table:
                self.symbol_table[self.token_buffer] = [len(self.symbol_table) + 1]
        return self.token_buffer

    def get_next_token(self) -> Tuple:
        """Return next token of input_file. None if EOF."""
        self.token_buffer = ""
        self.current_state = self.states[0]
        while True:
            """ Terminal state"""
            if self.current_state.is_terminal:
                if self.current_state.is_lookahead:
                    self.token_buffer = self.token_buffer[:-1]
                    self.forward_step_back()
                return self.token_tuple()

            """ Non-terminal state """
            self.current_char = self.get_next_char()
            # EOF
            if self.current_char is None:
                if self.current_state.number in {1, 3, 6, 9, 11}:
                    self.forward_step_back()
                    return self.token_tuple()
                self.handle_error(Error.INCOMPLETE_TOKEN)
                return None

            # not EOF
            self.token_buffer += self.current_char
            """ Choosing appropriate transition """
            for transition in self.current_state.out_transitions:
                if self.current_char in transition.charset:
                    self.current_state = transition.dest
                    break
            else:  # No appropriate transition found -> Error
                self.handle_error(Error.NO_TRANSITION)

    def forward_step_back(self):
        """ Move <forward> backward """
        if self.forward == 0:
            self.forward = 2 * self.BUFFER_SIZE - 1
        else:
            self.forward -= 1

    def get_next_char(self) -> str:
        """Return next character of input_file. None if EOF."""
        # check if buffer needs to be reloaded
        if self.forward == 0:
            self.buffer[:self.BUFFER_SIZE] = self._load_buffer()
        elif self.forward == self.BUFFER_SIZE:
            self.buffer[self.BUFFER_SIZE:] = self._load_buffer()
        char = self.buffer[self.forward]

        # update forward
        self.forward += 1
        if self.forward == 2 * self.BUFFER_SIZE:
            self.forward = 0

        return char

    def _load_buffer(self) -> List[Optional[str]]:
        """Return a list of characters of length BUFFER_SIZE.

        The characters are read from input_file.
        """
        temp = self.input_file.read(self.BUFFER_SIZE)
        return [temp[i] if i < len(temp) else None for i in range(self.BUFFER_SIZE)]

    def _initialize_symbol_table(self):
        for keyword in Scanner.keywords:
            self.symbol_table[keyword] = [len(self.symbol_table) + 1]

    def _initialize_states(self):
        """Creates states and transitions"""
        # create states
        self.states: List[State] = [State(i) for i in range(19)]

        # set terminal states
        for i in (2, 4, 5, 7, 8, 10, 12, 15, 16, 18):
            self.states[i].set_terminal()

        # set lookahead states
        for i in (2, 4, 8, 10, 12, 18):
            self.states[i].set_lookahead()

        # add transitions
        self.states[0].add_transition(Transition(self.states[0], self.states[1], self.digits))
        self.states[0].add_transition(Transition(self.states[0], self.states[3], self.letters))
        self.states[0].add_transition(Transition(self.states[0], self.states[5], self.symbols - {'/', '=', '*'}))
        self.states[0].add_transition(Transition(self.states[0], self.states[6], {'='}))
        self.states[0].add_transition(Transition(self.states[0], self.states[9], {'/'}))
        self.states[0].add_transition(Transition(self.states[0], self.states[16], self.whitespaces))
        self.states[0].add_transition(Transition(self.states[0], self.states[17], {'*'}))
        self.states[1].add_transition(Transition(self.states[1], self.states[1], self.digits))
        self.states[1].add_transition(Transition(self.states[1], self.states[2], self.symbols.union(self.whitespaces)))
        self.states[3].add_transition(Transition(self.states[3], self.states[3], self.alphanumerics))
        self.states[3].add_transition(Transition(self.states[3], self.states[4], self.symbols.union(self.whitespaces)))
        self.states[6].add_transition(Transition(self.states[6], self.states[7], {'='}))
        self.states[6].add_transition(Transition(self.states[6], self.states[8], self.valid_chars - {'='}))
        self.states[9].add_transition(Transition(self.states[9], self.states[10], self.valid_chars - {'*', '/'}))
        self.states[9].add_transition(Transition(self.states[9], self.states[11], {'/'}))
        self.states[9].add_transition(Transition(self.states[9], self.states[13], {'*'}))
        self.states[11].add_transition(Transition(self.states[11], self.states[11], self.all_chars - {'\n'}))
        self.states[11].add_transition(Transition(self.states[11], self.states[12], {'\n', self.EOF}))
        self.states[13].add_transition(Transition(self.states[13], self.states[13], self.all_chars - {'*'}))
        self.states[13].add_transition(Transition(self.states[13], self.states[14], {'*'}))
        self.states[14].add_transition(Transition(self.states[14], self.states[15], {'/'}))
        self.states[17].add_transition(Transition(self.states[18], self.states[17], self.all_chars - {'/'}))


if __name__ == '__main__':
    scanner = Scanner()
    while True:
        current_token = scanner.get_next_token()
        if current_token is None:
            print("current token: ", "EOF")
            print("Program Finished")
            break
        print("current token: ", current_token)
    if scanner.error_count == 0:
        scanner.error_file.write("There is no lexical error.")
    for key in scanner.symbol_table:
        scanner.symbol_table_file.write(str(scanner.symbol_table[key]) + ": " + key + "\n")