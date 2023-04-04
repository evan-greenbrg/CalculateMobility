#!/bin/bash
river='Jurua_Gaviao'
poly="/Volumes/Greenberg/Moility/Comparative/SingleRivers/Jurua_Gaviao/$river.gpkg"
metrics="single"
# metrics="single"
# out="/Users/greenberg/Documents/PHD/Writing/MobilityMethods/Figures/PubFigures/Figure6_Avulsion/$river"
out="/Volumes/Greenberg/Moility/Comparative/SingleRivers/Jurua_Gaviao/WaterLevel1"

python ../CalculateMobility/main.py --poly $poly --metrics $metrics --out $out --river $river

