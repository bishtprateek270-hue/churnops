"""
Unit tests for centralized configuration module (src/config.py).
"""

import os
import unittest
from unittest.mock import patch

from src.config import Settings


class TestConfigSettings(unittest.TestCase):
    """Test suite for environment-aware Settings dataclass."""

    def test_default_settings(self):
        """Test default fallback settings."""
        settings = Settings()
        self.assertEqual(settings.MLFLOW_EXPERIMENT_NAME, "ChurnOps_Churn_Prediction")
        self.assertEqual(settings.MODEL_NAME, "ChurnOps-Model")
        self.assertEqual(settings.TEST_SIZE, 0.2)
        self.assertEqual(settings.RANDOM_STATE, 42)
        self.assertEqual(settings.FAST_MODE_TRIALS, 5)
        self.assertEqual(settings.COST_FN, 500.0)
        self.assertEqual(settings.COST_FP, 50.0)

    def test_environment_override(self):
        """Test overriding settings via environment variables."""
        with patch.dict(os.environ, {
            "MLFLOW_EXPERIMENT_NAME": "Custom_Experiment",
            "MODEL_NAME": "Custom-Model-Name",
            "COST_FN": "750.0",
            "FAST_MODE_TRIALS": "10",
        }):
            custom_settings = Settings()
            self.assertEqual(custom_settings.MLFLOW_EXPERIMENT_NAME, "Custom_Experiment")
            self.assertEqual(custom_settings.MODEL_NAME, "Custom-Model-Name")
            self.assertEqual(custom_settings.COST_FN, 750.0)
            self.assertEqual(custom_settings.FAST_MODE_TRIALS, 10)


if __name__ == "__main__":
    unittest.main()
