% Files
%   computeFandPhi               - Compute flow-capacity/storage-capacity diagram (F,Phi)
%   computeLorenz                - Compute the Lorenz coefficient
%   computeSweep                 - Compute sweep efficiency versus dimensionless time (PVI)
%   computeTOFandTracer          - Compute time-of-flight and tracer distribution using finite-volume scheme.
%   computeTOFandTracerAverage   - Executes computeTOFandTracer for a series of states and averages
%   computeTimeOfFlight          - Compute time of flight using finite-volume scheme.
%   computeWellPairs             - Compute volumes and fluxes associated with each flux pair
%   expandCoarseWellCompletions  - Pseudo-wells for computing flow diagnostics in an upscaled model
%   expandWellCompletions        - Pseudo-wells for computation of flow diagnostics for completions
%   interactiveDiagnostics       - Launch an interactive diagnostics session
%   plotTOFArrival               - 
%   plotTracerBlend              - Visualise tracer partitions: gray regions are affected by multiple tracers
%   plotWellAllocationComparison - Plot a panel comparing well-allocation from models with different resolution
%   plotWellAllocationPanel      - Plot a panel comparing well-allocation from models with different resolution
%   plotWellPairConnections      - Plot lines between wells to show relative flux allocation
%   selectTOFRegion              - Select a subset of cells based on TOF criteria
%   validateStateForDiagnostics  - Validate and fix state for flow diagnostics

%{
Copyright 2009-2018 SINTEF ICT, Applied Mathematics.

This file is part of The MATLAB Reservoir Simulation Toolbox (MRST).

MRST is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

MRST is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with MRST.  If not, see <http://www.gnu.org/licenses/>.
%}
