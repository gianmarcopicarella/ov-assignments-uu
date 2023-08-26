If the preprocess variable SIMD is defined (line 42 Game.cpp), then the SIMD version of Game::Simulation is used.
Otherwise, the GPU version of Game::Simulation is used.

If the preprocess variable PROFILE is defined (line 41 Game.cpp), then the two loops in Game::Simulation are profiled and the average execution time in nanoseconds is printed. 
By default, the number of iterations is set to 1000.