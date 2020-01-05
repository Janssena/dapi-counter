# dapi-counter

This repository covers the efforts to create a cell counting model that can effectively predict the number of DAPI stained cells in a Z-stack of images taken with a fluorescence microscope.
The model consists of three components:

# mask
This component takes in each image of the Z-stack and returns a black and white mask image.

# count 
This component takes the produced mask image as an input and returns the number of cells as an Integer, and the cell locations as an array of Integers.

# z-stack
This component takes the locations of the cells in each image in the Z-stack and resolves the cell count in the complete volume.
It does so by detecting count artifacts and using a recurrent setup where information about the cell locations from the previous slice are used to validate the cell locations in the next slice.
If a particular location is present in multiple slices its chance of becoming a true cell increases.
