# Mohammad Mahdi Sadeghi    99105548
# Benyamin Maleki           99102286

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
        self._parser.run()
        self._parser.save_parse_tree()
        self._parser.save_errors()


if __name__ == '__main__':
    compiler = Compiler()
    compiler.run()
