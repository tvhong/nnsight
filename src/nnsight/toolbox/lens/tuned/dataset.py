
import zstandard
import io

import jsonlines
import simdjson as json

parser = json.Parser()

def json_parser(x):
    try:
        line = parser.parse(x).as_dict()
        return line
    except ValueError:
        return x

class PileReader:
    def __init__(self, filenames, para_joiner='\n\n'):
        if not isinstance(filenames, list):
            filenames = [filenames]
        self.filenames = filenames
        self.para_joiner = para_joiner

    def _read_fn(self, filename):
        with open(filename, 'rb+') as f:
            cctx = zstandard.ZstdDecompressor()
            reader_stream = io.BufferedReader(cctx.stream_reader(f))
            reader = jsonlines.Reader(reader_stream, loads=json_parser)
            for item in reader:
                result = dict()
                if isinstance(item, str):
                    result['text'] = item
                else:
                    text = item['text']
                    if isinstance(text, list):
                        text = self.para_joiner.join(text)
                        result['text'] = text
                yield result
    
    def __iter__(self):
        for filename in self.filenames:
            return self._read_fn(filename)

