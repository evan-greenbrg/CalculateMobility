#/bin/bash
river='Amur_Khabarovsk'
poly="/Volumes/Greenberg/Moility/ComparativeMobility/DataArchive/Mobility/B/Working/Amur_Khabarovsk/$river.gpkg"
metrics="single"
stop=30
# metrics="single"
# out="/Users/greenberg/Documents/PHD/Writing/MobilityMethods/Figures/PubFigures/Figure6_Avulsion/$river"
# out="/Volumes/Greenberg/Moility/ComparativeMobility/Channel_Belts/Rivers/$river"
out="/Volumes/Greenberg/Moility/ComparativeMobility/DataArchive/Mobility/B/Working/Amur_Khabarovsk/$river"

python ../CalculateMobility/main.py --poly $poly --stop $stop --metrics $metrics --out $out --river $river

