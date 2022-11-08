#!/bin/bash
for value in Amazonas_Jatuarana Amazonas_Obidos Guapore_PrincipeDeBeira Irtysh_Tobolsk Lena_Tabaga Ob_Prokhorkino Orinoco_Musinacio Yukon_NearStevensVillage Zambezi_Senanga
do
    echo $value
    river=$value
    poly="/Users/greenberg/Documents/PHD/Projects/Mobility/MethodsPaper/RiverData/AustinMasks/RePull/$value/$value.gpkg"
    gif="true"
    out="/Users/greenberg/Documents/PHD/Projects/Mobility/MethodsPaper/RiverData/AustinMasks/RePull"
    scale=30

    python ../CalculateMobility/main.py --poly $poly --gif $gif --out $out --river $river --scale $scale

done

