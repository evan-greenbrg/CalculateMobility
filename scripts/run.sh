#!/bin/bash

river='Width5'
poly="/Users/greenberg/Documents/PHD/Projects/Mobility/MethodsPaper/SensitivityTest/BermejoWidths/$river.gpkg"
gif="true"
out="/Volumes/Samsung_T5/Mac/PhD/Projects/Mobility/MethodsPaper/RiverData/BermajoWidths"
scale=30

python ../CalculateMobility/main.py --poly $poly --gif $gif --out $out --river $river --scale $scale
