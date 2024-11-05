import unittest
import os
from amcreport import ExamData
import sqlite3

class TestReport(unittest.TestCase):

    def setUp(self):
        self.path = os.getcwd()
        print(self.path)
        self.exam_data = ExamData(self.path)

    def test_init(self):
        self.assertEqual(self.exam_data.path, self.path)
        self.assertEqual(self.exam_data.threshold, 99)

    def test_check_db(self):
        # Test if the databases exist
        self.assertTrue(os.path.exists(self.exam_data.scoring_db))
        self.assertTrue(os.path.exists(self.exam_data.capture_db))

    def test_get_student_code_length(self):
        # Test if the student code length is correctly retrieved
        self.assertIsInstance(self.exam_data.scl, int)

    def test_get_marks(self):
        # Test if the marks are correctly retrieved from the database
        self.assertIsInstance(self.exam_data.marks, pd.DataFrame)

    def test_get_scores(self):
        # Test if the scores are correctly retrieved from the database
        self.assertIsInstance(self.exam_data.scores, pd.DataFrame)

    def test_get_questions(self):
        # Test if the questions are correctly retrieved from the database
        self.assertIsInstance(self.exam_data.questions, pd.DataFrame)

    def test_get_items(self):
        # Test if the items are correctly retrieved from the database
        self.assertIsInstance(self.exam_data.items, pd.DataFrame)

    def test_general_stats(self):
        # Test if the general statistics are correctly calculated
        self.assertIsInstance(self.exam_data.general_stats, pd.DataFrame)

    def test_ticked(self):
        # Test if the ticked method correctly determines if an answer box has been ticked
        # This test requires a specific row from the capture database
        # You may need to modify this test based on your actual data
        conn = sqlite3.connect(self.exam_data.capture_db)
        row = conn.execute("SELECT * FROM capture LIMIT 1").fetchone()
        conn.close()
        self.assertIsInstance(self.exam_data._ticked(row), int)

def main() -> None:
    unittest.main()


if __name__ == '__main__':
    main()