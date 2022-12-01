#!/bin/bash
river='Araguaia'
poly="/Users/greenberg/Documents/PHD/Projects/Mobility/MethodsPaper/RiverData/Meandering/shapes/$river.gpkg"
metrics="dswe"
out="/Volumes/Samsung_T5/Mac/PhD/Projects/Mobility/MethodsPaper/RiverData/MeanderingRivers/Data/$river"

python ../CalculateMobility/main.py --poly $poly --metrics $metrics --out $out --river $river
