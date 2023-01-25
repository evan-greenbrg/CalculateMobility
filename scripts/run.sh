#!/bin/bash
river='Bermejo'
poly="/Users/greenberg/Documents/PHD/Projects/Mobility/MethodsPaper/SensitivityTest/Bermejo/$river.gpkg"
metrics="dswe"
out="/Users/greenberg/Documents/PHD/Projects/Mobility/MethodsPaper/SensitivityTest/Bermejo/$river"

python ../CalculateMobility/main.py --poly $poly --metrics $metrics --out $out --river $river
