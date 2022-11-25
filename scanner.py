import string
from typing import List, Optional, Set


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

    def __init__(self):
        """Inits Scanner"""
        self.input_file = open("input.txt", mode="r")
        self.buffer: List[Optional[str]] = []
        self.forward: int = 0

        self._initialize_states()

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

    def _initialize_states(self):
        """Creates states and transitions"""
        # create states
        self.states: List[State] = [State(i) for i in range(17)]

        # set terminal states
        for i in (2, 4, 5, 7, 8, 10, 12, 15, 16):
            self.states[i].set_terminal()

        # set lookahead states
        for i in (2, 4, 8, 10, 12):
            self.states[i].set_lookahead()

        # add transitions
        self.states[0].add_transition(Transition(self.states[0], self.states[1], self.digits))
        self.states[0].add_transition(Transition(self.states[0], self.states[3], self.letters))
        self.states[0].add_transition(Transition(self.states[0], self.states[5], self.symbols - {'/', '='}))
        self.states[0].add_transition(Transition(self.states[0], self.states[6], {'='}))
        self.states[0].add_transition(Transition(self.states[0], self.states[9], {'/'}))
        self.states[0].add_transition(Transition(self.states[0], self.states[16], self.whitespaces))
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


if __name__ == '__main__':
    scanner = Scanner()
    while True:
        current_char = scanner.get_next_char()
        print("current char: ", current_char)
        if current_char is None:
            print("EOF")
            break
