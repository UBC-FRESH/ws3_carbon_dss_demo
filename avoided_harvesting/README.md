This directory contains a "avoided harvesting" case for the DSS. 

The case includes three Jupyter notebooks.

Notebook `00_woodstock_files` compiles Woodstock input files from raw input data. We use the Woodstock input files to build the `ws3.forest.ForestModel` instances in the other two notebooks.

Notebook `01_ws3_demo` is a minimalist demo of building a ws3 model from the Woodstock input files, scheduling some harvest, and poking around the internal yield curve data structures. 

Notebook `02_dss` is the main DSS notebook. It simulates baseline and alternative scenarios in ws3, pushes these to CBM to calculate carbon indictors, and compiles and presents output.