#!/bin/bash

python -m main.train --config-file config/HDTF_TFHP.yaml 2>&1 | tee output.log