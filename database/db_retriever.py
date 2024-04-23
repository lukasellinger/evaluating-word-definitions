# Adapted from
# https://github.com/sheffieldnlp/fever-naacl-2018/blob/a322719/src/retrieval/fever_doc_db.py
# Originally taken from
# https://github.com/facebookresearch/DrQA/blob/master/drqa/retriever/doc_db.py
#
# Additional license and copyright information for this source code are available at:
# https://github.com/facebookresearch/DrQA/blob/master/LICENSE
# https://github.com/sheffieldnlp/fever-naacl-2018/blob/master/LICENSE
"""Documents, in a sqlite database."""

import sqlite3
import unicodedata

from config import PROJECT_DIR

class FeverDocDB:
    """Sqlite backed document storage."""

    def __init__(self, db_path=None):
        if not db_path:
            self.path = PROJECT_DIR.joinpath('database/fever.db')
        else:
            self.path = db_path
        self.connection = sqlite3.connect(self.path, check_same_thread=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def write(self, statement, params=()):
        """Write statement to database."""
        cursor = self.connection.cursor()
        cursor.execute(statement, params)
        cursor.close()
        self.connection.commit()

    def get_doc_lines(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT lines FROM documents WHERE document_id = ?", (doc_id,),
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]
