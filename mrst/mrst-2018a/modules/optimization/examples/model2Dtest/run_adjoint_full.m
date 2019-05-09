%% sensitivitiesModel2D - analyse sensitivity capabilities 
clear all;
LASTN = maxNumCompThreads(4);
mrstModule add ad-core ad-blackoil ad-props optimization spe10 mrst-gui

inputArg = getenv('case_name');  

% Setup model -> grid, rock, schedule, fluid etc
nxyz = [128, 64, 1];
Dxyz = [128*4, 64*4, 10];

G = cartGrid(nxyz, Dxyz);
G = computeGeometry(G);

rock = getGroundTruthModel(1:128, 1:64, 1);

% fluid
pRef = 200*barsa;

fluid = initSimpleADIFluid('mu',    [.3, 5, 0]*centi*poise, ...
                           'rho',   [1000, 700, 0]*kilogram/meter^3, ...
                           'n',     [2, 2, 0]);
c = 1e-5/barsa;
p_ref = 200*barsa;
fluid.bO = @(p) exp((p - p_ref)*c);

W = [];
% Injectors (lower-left and upper-right)
%ci(1) = 32*128+128-8
ci = [8, 136, 264, 392, 520, 648, 776, 904, 1032, 1160, 1288, 1416, 1544, 1672, 1800, 1928, 2056, 2184, 2312, 2440, 2568, 2696, 2824, 2952, 3080, 3208, 3336, 3464, 3592, 3720, 3848, 3976, 4104, 4232, 4360, 4488, 4616, 4744, 4872, 5000, 5128, 5256, 5384, 5512, 5640, 5768, 5896, 6024, 6152, 6280, 6408, 6536, 6664, 6792, 6920, 7048, 7176, 7304, 7432, 7560, 7688, 7816, 7944, 8072];
%ci(k)
for k  = 1:1
    W = addWell(W, G, rock, ci, 'Type' , 'rate', ...
                                   'Val'  , 300*meter^3/day, ...
                                   'Dir', 'x', ...
                                   'Name' , sprintf('I%d', k), ...
                                   'comp_i', [1 0], ...
                                   'Sign' , 1);
end

% Producers (upper-left and -right)
%cp(1) = 32*128+8;
cp = [120, 248, 376, 504, 632, 760, 888, 1016, 1144, 1272, 1400, 1528, 1656, 1784, 1912, 2040, 2168, 2296, 2424, 2552, 2680, 2808, 2936, 3064, 3192, 3320, 3448, 3576, 3704, 3832, 3960, 4088, 4216, 4344, 4472, 4600, 4728, 4856, 4984, 5112, 5240, 5368, 5496, 5624, 5752, 5880, 6008, 6136, 6264, 6392, 6520, 6648, 6776, 6904, 7032, 7160, 7288, 7416, 7544, 7672, 7800, 7928, 8056, 8184];
%cp(k)
for k  = 1:1
    W = addWell(W, G, rock, cp, 'Type', 'bhp', ...
                                   'Val' , 150*barsa, ...
                                   'Dir', 'x', ...
                                   'Name', sprintf('P%d', k), ...
                                   'comp_i', [0 1], ...
                                   'Sign', -1);
end

% Set up 4 control-steps each 150 days
ts = { [1 1 3 5 5 10 10 10 15 15 15 15 15 15 15]'*day, ...
        repmat(150/10, 10, 1)*day, ...
        repmat(150/6, 6, 1)*day, ...
        repmat(150/6, 6, 1)*day};
       
numCnt = numel(ts);

[schedule.control(1:numCnt).W] = deal(W);
schedule.step.control = rldecode((1:4)', cellfun(@numel, ts));
schedule.step.val     = vertcat(ts{:});

gravity off
%% Reset fluid to include scaling:
% $s_w -> \frac{s_w-swcr}{swu-swcr}$
% $s_o -> \frac{s_o-sowcr}{1-swl-sowcr}$
fluid = initSimpleScaledADIFluid('mu',    [.3, 5, 0]*centi*poise, ...
                                 'rho',   [1000, 700, 0]*kilogram/meter^3, ...
                                 'n',     [2, 2, 0], ...
                                 'swl',   0.10*ones(G.cells.num,1), ...
                                 'swcr',  0.15*ones(G.cells.num,1), ...
                                 'sowcr', 0.12*ones(G.cells.num,1), ...
                                 'swu',   0.90*ones(G.cells.num,1));
                                 
                       
% Create model-object of class TwoPhaseOilWaterModel  
model_ref  = load(join(['utils/', inputArg, '/model_ref.mat']));%                   
model_ref = model_ref.model_ref;

% Set initial state and run simulation:
state0 = initResSol(G, 200*barsa, [.15, .85]);

% Set up a perturbed model with different pv and perm:
rock1 = gethalfcircle();

model = TwoPhaseOilWaterModel(G, rock1, fluid);

% load ref model
ws_ref = load(join(['utils/', inputArg, '/ws_ref.mat']));
ws_ref = ws_ref.ws_ref;

states_ref = load(join(['utils/', inputArg, '/states_ref.mat']));
states_ref = states_ref.states_ref;

r_ref = load(join(['utils/', inputArg, '/r_ref.mat']));
r_ref = r_ref.r_ref;

% run model
[ws, states, r] = simulateScheduleAD(state0, model, schedule);

save('utils/synthetic/ws.mat', 'ws');
