#!/bin/bash
poly="/home/greenberg/ExtraSpace/PhD/Projects/Mobility/MethodsPaper/LowSin/shapes/Charysh_Karpovo2.gpkg"
gif="true"
out="/home/greenberg/ExtraSpace/PhD/Projects/Mobility/MethodsPaper/LowSin/files"
river="Charysh_Karpovo2"

python ../CalculateMobility/main.py --poly $poly --gif $gif --out $out --river $river
