#!/bin/bash
river='Ucayali_Requena'
poly="/Volumes/Greenberg/Moility/GalleazziData/shapes/$river.gpkg"
metrics="dswe"
# metrics="single"
# out="/Users/greenberg/Documents/PHD/Writing/MobilityMethods/Figures/PubFigures/Figure6_Avulsion/$river"
out="/Volumes/Greenberg/Moility/GalleazziData/RiverData/$river"

python ../CalculateMobility/main.py --poly $poly --metrics $metrics --out $out --river $river

for value in 1 2 3 4
do
    echo $value
    gif_root="$out/WaterLevel$value/mask"
    gif_out="$out/WaterLevel$value.gif"
    python make_gif.py --root $gif_root --out $gif_out
done
