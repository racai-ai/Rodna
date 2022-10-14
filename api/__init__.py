from pathlib import Path
from conllu.models import SentenceList

class ConlluProcessor(object):
    """Introduces a super class of text processing API,
    to facilitate Rodna vs. competition comparison."""

    def process_text_file(self, txt_file: str):
        """Takes a UTF-8 Romanian text file, processes it with Rodna and
        outputs the .conllu file in the same folder."""

        with open(txt_file, mode='r', encoding='utf-8') as f:
            all_text_lines = ''.join(f.readlines())
        # end with

        sentences = self.process_text(text=all_text_lines)
        txt_file_path = Path(txt_file)
        processor_name = self.__class__.__name__.lower()

        if processor_name.endswith('processor'):
            processor_name = processor_name.replace('processor', '')
        # end if

        if not processor_name:
            processor_name = 'generic'
        # end if

        cnl_file_path = txt_file_path.parent / \
            Path(txt_file_path.stem + '.' + processor_name + '.conllu')

        # Print the CoNLL-U file
        with open(cnl_file_path, mode='w', encoding='utf-8') as f:
            for token_list in sentences:
                print(token_list.serialize(), file=f, end='')
            # end for
        # end with

    def process_text(self, text: str) -> SentenceList:
        """To be implemented in subclasses."""
        pass
