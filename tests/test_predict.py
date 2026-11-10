import unittest
from contextlib import redirect_stdout
from io import StringIO

from predict import predict, validate_input


class RecordingVectorizer:
    def __init__(self):
        self.transformed_texts = None

    def transform(self, texts):
        self.transformed_texts = texts
        return texts


class FixedClassifier:
    classes_ = ["Good Fit", "No Fit", "Potential Fit"]

    def predict(self, features):
        return ["Potential Fit"]

    def predict_proba(self, features):
        return [[0.1, 0.2, 0.7]]


class TestPredictInputValidation(unittest.TestCase):
    def setUp(self):
        self.vectorizer = RecordingVectorizer()
        self.model = FixedClassifier()

    def test_valid_inputs_keep_existing_prediction_contract(self):
        result = predict(
            '  PYTHON "Developer"\n',
            " Seeking Python Engineer ",
            self.vectorizer,
            self.model,
        )

        self.assertEqual(
            self.vectorizer.transformed_texts,
            ["python developer seeking python engineer"],
        )
        self.assertEqual(
            result,
            {
                "label": "Potential Fit",
                "probs": {
                    "Good Fit": 0.1,
                    "No Fit": 0.2,
                    "Potential Fit": 0.7,
                },
            },
        )

    def test_invalid_resume_is_rejected_before_vectorization(self):
        invalid_inputs = ["", " \t\n ", None, 123]

        for value in invalid_inputs:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "Resume"):
                    predict(value, "Valid job description", self.vectorizer, self.model)
                self.assertIsNone(self.vectorizer.transformed_texts)

    def test_invalid_job_description_is_rejected_before_vectorization(self):
        invalid_inputs = ["", " \t\n ", None, 123]

        for value in invalid_inputs:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "Job description"):
                    predict("Valid resume", value, self.vectorizer, self.model)
                self.assertIsNone(self.vectorizer.transformed_texts)

    def test_cli_empty_input_keeps_print_and_exit_behavior(self):
        output = StringIO()
        with redirect_stdout(output):
            with self.assertRaises(SystemExit) as error:
                validate_input(" \t", "Resume")

        self.assertEqual(error.exception.code, 1)
        self.assertIn(
            "[ERROR] Resume cannot be empty. Please provide some text.",
            output.getvalue(),
        )

    def test_cli_non_string_input_exits_with_clear_message(self):
        output = StringIO()
        with redirect_stdout(output):
            with self.assertRaises(SystemExit) as error:
                validate_input(None, "Resume")

        self.assertEqual(error.exception.code, 1)
        self.assertIn("[ERROR] Resume must be a string.", output.getvalue())


if __name__ == "__main__":
    unittest.main()
