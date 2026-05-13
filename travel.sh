#!/bin/bash
cd "$(dirname "$0")"
source taenv/bin/activate
python run_rich.py "$@"
