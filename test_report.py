import unittest
from report import get_label

class TestGetLabelFunction(unittest.TestCase):

    def test_difficulty_label(self):
        # Test difficulty label for different values
        self.assertEqual(get_label('difficulty', 0.3), ('Difficult', 'red'))
        self.assertEqual(get_label('difficulty', 0.5), ('Intermediate', 'yellow'))
        self.assertEqual(get_label('difficulty', 0.8), ('Easy', 'green'))

    def test_discrimination_label(self):
        # Test discrimination label for different values
        self.assertEqual(get_label('discrimination', -0.1), ('Review!', 'red'))
        self.assertEqual(get_label('discrimination', 0.1), ('Low', 'grey'))
        self.assertEqual(get_label('discrimination', 0.2), ('Moderate', 'yellow'))
        self.assertEqual(get_label('discrimination', 0.4), ('High', 'green'))
        self.assertEqual(get_label('discrimination', 0.6), ('Very high', 'blue'))

    def test_correlation_label(self):
        # Test correlation label for different values
        self.assertEqual(get_label('correlation', -0.1), ('Review!', 'red'))
        self.assertEqual(get_label('correlation', 0.05), ('None', 'white'))
        self.assertEqual(get_label('correlation', 0.15), ('Low', 'grey'))
        self.assertEqual(get_label('correlation', 0.25), ('Moderate', 'yellow'))
        self.assertEqual(get_label('correlation', 0.4), ('Strong', 'green'))
        self.assertEqual(get_label('correlation', 0.6), ('Very strong', 'blue'))

    def test_invalid_column(self):
        # Test for invalid column name
        self.assertEqual(get_label('invalid_column', 0.5), ('-', 'white'))

if __name__ == '__main__':
    unittest.main()