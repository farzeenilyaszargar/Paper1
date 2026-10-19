import unittest
from collections import Counter

from models.character_exposure import compare_characters
from models.representations import CHAR_IDS


class CharacterExposureTests(unittest.TestCase):
    def test_counts_occurrences_not_only_distinct_characters(self):
        result = compare_characters(Counter("p000"),Counter("p111 " ))
        self.assertEqual(result["source_unseen_characters"],[" ","1"])
        self.assertEqual(result["source_unseen_character_ids"],[CHAR_IDS[" "],CHAR_IDS["1"]])
        self.assertEqual(result["unseen_occurrences"],4)
        self.assertEqual(result["total_content_tokens"],5)
        self.assertEqual(result["unseen_fraction"],.8)
        self.assertEqual(compare_characters(Counter("p012"),Counter("p111"))["unseen_fraction"],0)

    def test_invalid_counts_and_unshared_alphabet_rejected(self):
        for source,target in (({},{}),({"p":0},{"p":1}),({"p":1},{"p":-1}),({"p":1},{"é":1})):
            with self.assertRaises(ValueError):
                compare_characters(source,target)


if __name__ == "__main__":
    unittest.main()
