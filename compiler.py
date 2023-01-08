# Mohammad Mahdi Sadeghi    99105548
# Benyamin Maleki           99102286

from typing import Dict, List, Optional, Tuple

from c_parser import Parser
from scanner import Scanner


class Compiler:
    """
    C-Minus compiler.

    Attributes:
        scanner the compiler's scanner
    """
    def __init__(self):
        """Inits Compiler"""
        self._scanner = Scanner()
        self._parser = Parser(self._scanner)

    def run(self):
        """Runs the compiler and compiles input.txt."""
        res = self._parser.run()
        if res is None:
            print("EOF reached in panic mode!")
        # tokens_dict: Dict[int, List[Optional[Tuple[str, str]]]] = {}
        # while True:
        #     current_token = self._scanner.get_next_token()
        #     if current_token[0] == self._scanner.EOF:
        #         break
        #     if self._scanner.line_number in tokens_dict:
        #         tokens_dict[self._scanner.line_number].append(current_token)
        #     else:
        #         tokens_dict[self._scanner.line_number] = [current_token]
        #
        # tokens_file = open("tokens.txt", "w")
        # for line_num in sorted(tokens_dict.keys()):
        #     line = ''.join([f"({token[0]}, {token[1]}) " for token in tokens_dict[line_num]])
        #     tokens_file.write(f"{line_num}.\t{line}\n")
        #
        # self._scanner.save_errors()
        # self._scanner.save_symbols()


if __name__ == '__main__':
    compiler = Compiler()
    compiler.run()
