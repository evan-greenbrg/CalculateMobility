#!/bin/bash
river='Trinity'
poly="/Users/greenberg/Documents/PHD/Writing/MobilityMethods/Figures/PubFigures/Figure4_Meandering/Trinity/$river.gpkg"
metrics="dswe"
out="/Users/greenberg/Documents/PHD/Writing/MobilityMethods/Figures/PubFigures/Figure4_Meandering/Trinity/$river"

python ../CalculateMobility/main.py --poly $poly --metrics $metrics --out $out --river $river
