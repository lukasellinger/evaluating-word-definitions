"""Module for reading files."""
import json


class Reader:
    """General file reader."""

    def __init__(self, encoding="utf-8"):
        self.enc = encoding

    def read(self, file):
        """Read a file."""
        with open(file, "r", encoding=self.enc) as f:
            return self.process(f)

    def write(self, file, lines):
        """Write lines to file."""
        with open(file, "a", encoding=self.enc) as f:
            self._write(f, lines)

    def _write(self, file, lines):
        """Write lines to an opened file."""

    def process(self, file):
        """Process an opened file."""


class JSONLineReader(Reader):
    """Reader for .jsonl files."""

    def process(self, file):
        """Read each line as json object."""
        data = []
        for line in file.readlines():
            data.append(json.loads(line.strip()))

        return data

    def _write(self, file, lines):
        for line in lines:
            json.dump(line, file)
            file.write('\n')


class JSONReader(Reader):
    """Reader for .json files."""

    def process(self, file):
        """Read file as json object."""
        return json.load(file)
