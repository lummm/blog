#!/bin/bash

jupyter nbconvert ./notebooks/*.ipynb  --to markdown --output-dir=./notebook_output
