#!/usr/bin/env bash

set -e

python crawler.py
python src/ingestion.py
