#!/bin/bash
river='PearlUpstream'
poly="/home/greenberg/ExtraSpace/PhD/Projects/Mobility/Dams/River_Shapes/$river.gpkg"
gif="true"
out="/home/greenberg/ExtraSpace/PhD/Projects/Mobility/Dams/River_Files"
scale=30

python ../CalculateMobility/main.py --poly $poly --gif $gif --out $out --river $river --scale $scale
