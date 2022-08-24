#!/bin/bash
poly="/home/greenberg/ExtraSpace/PhD/Projects/Mobility/MethodsPaper/LowSin/shapes/Araguaia_LuizAlves.gpkg"
gif="true"
out="/home/greenberg/ExtraSpace/PhD/Projects/Mobility/MethodsPaper/LowSin/files"
river="Araguaia_LuizAlves"

python ../CalculateMobility/main.py --poly $poly --gif $gif --out $out --river $river
