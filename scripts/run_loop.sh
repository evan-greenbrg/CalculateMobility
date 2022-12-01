#!/bin/bash
# for value in Amazon Araguaia Beni Branco Castelli1 Coco Curuca Huallaga Indus Jurua Kazak2 Madre_de_Dios Mamore Mississippi Niger PNG10 PNG13 PNG6 Purus Putamayo RedRiver RedUpstream Sacramento Sembakung Solimoes Strickland Ucayali Xingu
for value in Amazon 
do
    echo $value
    river=$value
    poly="/Users/greenberg/Documents/PHD/Projects/Mobility/MethodsPaper/RiverData/Meandering/shapes/$value.gpkg"
    metrics="dswe"
    out="/Users/greenberg/Documents/PHD/Projects/Mobility/MethodsPaper/RiverData/Meandering/files"

    python ../CalculateMobility/main.py --poly $poly --metrics $metrics --out $out --river $river --scale $scale

done



