#!/bin/bash

echo Running: python3 py/sham.py MW1 0.5 -q &
python3 py/sham.py MW1 0.5 -q &

echo Running: python3 py/sham.py MW1 0.6 -q
python3 py/sham.py MW1 0.6 -q

echo Running: python3 py/sham.py Mz 0.5 -q &
python3 py/sham.py Mz 0.5 -q &

echo Running: python3 py/sham.py Mz 0.6 -q
python3 py/sham.py Mz 0.6 -q
