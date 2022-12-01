#!/bin/bash
river='Sabine'
poly="/home/greenberg/ExtraSpace/PhD/Projects/Mobility/Dams/Rivers/Shapes/$river.gpkg"
metrics="dswe"
out="/home/greenberg/ExtraSpace/PhD/Projects/Mobility/Dams/Rivers/Files/$river"

python ../CalculateMobility/main.py --poly $poly --metrics $metrics --out $out --river $river
