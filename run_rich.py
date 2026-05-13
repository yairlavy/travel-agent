"""Entry point for the Rich terminal UI. Run: python run_rich.py"""
import warnings
import logging

warnings.warn = lambda *args, **kwargs: None   # silence all third-party warnings
logging.disable(logging.CRITICAL)              # silence all log output

from src.main_rich import run

if __name__ == "__main__":
    run()
