
import py_pdf_parser as ppp
import py_pdf_parser.loaders
import io
import base64

# 76 base64 chars per line
# 57 bytes per line
# 64 lines per page for pages 2 through 75
# plus one line on page 1
# plus 31 lines on page 76
# total of 72 * 64 + 32 = 4640 lines
# which means 264480 bytes
# but the last line is only 48 characters / 36 bytes long (instead of 57 bytes)
# so minus 21 for 264459 bytes

class FakeFile(io.StringIO):
    lines = [None] * 4640

    curpos = 0

    def tell(self) -> int:
        return self.curpos

    def seek(self, offset: int, whence: int = 0) -> None:
        print(f'{offset=} {whence=}')

        if whence == 0:
            actual_offset = 0 + offset
        elif whence == 1:
            actual_offset = curpos + offset
        elif whence == 2:
            actual_offset = 264459 + offset

        if actual_offset < 0:
            raise ValueError(f'Tried to seek below 0')
        if actual_offset > 264459:
            raise ValueError(f'Tried to seek after end of file')

        curpos = actual_offset

    def read(self, sz: int) -> bytes:
        line = self.curpos // 57
        offset_within_line = self.curpos - (line * 57)

        print(f'Reading from {self.curpos=} a length of {sz=} -- this results in {line=} and {offset_within_line=}')

        if not self.lines[line]:
            while not (the_line := self.try_read_line(line)):
                continue
            self.lines[line] = the_line

        return self.lines[line][offset_within_line : sz]

        raise NotImplementedError(f'Tried to read {sz} bytes from {self.curpos=}')

    def try_read_line(self, line: int) -> bytes | None:
        if line == 0:
            page = 1
            line_within_page = 0
        else:
            page = (line - 1) // 64 + 2
            line_within_page = (line - 1) - ((page - 2) * 64)

        # prompt the user to enter the line of text
        text = input(f'Hello, please enter line {line} of base64 -- you can find it on page {page}, line {line_within_page}\n> ')
        if len(text) != 76:
            print('Bad length - received {len(text)} characters instead of 76')
            return None
        print(f'Received {text}')
        return base64.b64decode(text)


ff = FakeFile()
py_pdf_parser.loaders.load(ff)
