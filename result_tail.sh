#!/bin/bash
cat "Data/$1/result.csv" | tail -n 20 | column -t -s ','
