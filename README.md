This is a time-dependent model of Stromboli volcano, aiming to capture first order dynamics of magma flow and thermal flow. 

There are three elements to the model, a deep reservoir with pressure P_deep, a conduit with bidirectional exchange flow of ascending hot volatile rich magma and descending crystal rich degassed cooler magma and a shallow reservoir where cooling and crystallisaton of magma takes place.

There are 5 ODEs, with a state vector made of P_deep, shallow reservoir mass, and temperatures of core flow, annulus flow and shallow reservoir. 

Params.py contains all the variables for the model and coupled_all_driver.py is the main program. This version of the code shows evolution to a steady state which approximates the reality of magma flow on Stromboli.

This code is part of a paper which is under development.

To run the code, produce an environmwnt with python 3.10 and Scipy, numpy, pandas and matplotlib installed.
