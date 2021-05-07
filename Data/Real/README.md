# Real

This directory contains the real data obtained from the Urban Observatory.

## File descriptions

The MCMC code requires several input files as described below. Note that here we use the C(++) style numbering convention and start counting from zero.

* radar\_data\_file.csv - the raw (not transformed) radar data in units of mm/h. Row i, column t contains the radar observation at grid location i at time t. Grid location zero is defined to be the top left corner and proceeds column-wise from thereon.

* gauge\_data\_file.csv - the raw (not transformed) gauge data in units of mm/h. Row i, column t contains the observation from gauge i at time t.

* gauge\_locations.csv - the grid locations of each rain gauge. Row i contains the location of rain gauge i.

* radar\_no\_censored.csv - the number of censored (zero) observations within the radar data file.

* gauge\_no\_censored.csv - the number of censored (zero) observations within the gauge data file.

* radar\_censored\_locations.csv - the locations of the censored observations within the radar data file. The top left corner of this file is considered to be location zero and the count proceeds column-wise from thereon.

* gauge\_censored\_locations.csv - the locations of the censored observations within the gauge data file. Again, the top left corner of this file is considered to be location zero and the count proceeds column-wise from thereon.

* neighbour\_location\_data.csv - the grid locations of the first-order neighbours. Row i contains 5 columns that correspond to the grid locations of the left neighbour, up neighbour, down neighbour, right neighbour and the current location, respectively.

* radar\_forecast\_data\_file.csv - this file is not required as an input to the MCMC code and contains the raw (not transformed) radar data at the next 6 time points (formatted as radar\_data\_file.csv)

* gauge\_forecast\_data\_file.csv - this file is not required as an input to the MCMC code and contains the raw (not transformed) gauge data at the next 6 time points (formatted as gauge\_data\_file.csv)

