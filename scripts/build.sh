#!/bin/bash

jupyter nbconvert ./notebooks/*.ipynb  --to markdown --output-dir=./_notebook_output
