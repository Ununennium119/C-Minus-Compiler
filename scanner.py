from typing import List, Optional


class Scanner:
    BUFFER_SIZE = 1024

    def __init__(self):
        self.input_file = open("input.txt", mode="r")
        self.buffer: List[Optional[str]] = []
        self.forward: int = 0

    def get_next_char(self) -> str:
        """Return next character of input_file. None if EOF"""
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


if __name__ == '__main__':
    scanner = Scanner()
    while True:
        current_char = scanner.get_next_char()
        print("current char: ", current_char)
        if current_char is None:
            print("EOF")
            break
