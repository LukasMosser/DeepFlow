runStandardModel('data/dissolutionTopSurfaceExample' , ...
                 @plotTopSurfaceExample              , ...
                 'dTi', 1 * year, 'dTm', 10 * year   , ...
                 'A' , [0 2]                         , ...
                 'residual', true                    , ...
                 'dis_types', {'none', 'rate'}       , ...
                 'fluid_types', {'sharp interface'   , ...
                                 'linear cap.'       , ...
                                 'P-scaled table'    , ...
                                 'P-K-scaled table'});
