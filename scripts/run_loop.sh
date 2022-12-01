#!/bin/bash
# for value in Amazon Araguaia Beni Branco Castelli1 Coco Curuca Huallaga Indus Jurua Kazak2 Madre_de_Dios Mamore Mississippi Niger PNG10 PNG13 PNG6 Purus Putamayo RedRiver RedUpstream Sacramento Sembakung Solimoes Strickland Ucayali Xingu
for value in Amazon 
do
    echo $value
    river=$value
    poly="/Users/greenberg/Documents/PHD/Projects/Mobility/MethodsPaper/RiverData/Meandering/shapes/$value.gpkg"
    gif="true"
    out="/Users/greenberg/Documents/PHD/Projects/Mobility/MethodsPaper/RiverData/Meandering/files"
    scale=30

    python ../CalculateMobility/main.py --poly $poly --gif $gif --out $out --river $river --scale $scale

done



