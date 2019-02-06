function rock = getGroundTruthModel(varargin)
   assert (all(cellfun(@isnumeric, varargin)), ...
           'Input arrays must be numeric.');

   % Load data in pre-processed form.
   rock = load_mat_file_gt();

   % Extract caller's requested subset from 'rock' data
   %
   % Return only PERM and PORO data and exclude any other information that
   % might be stored in on-disk representation of rock data.
   %
   ix   = define_subset(varargin{:});
   rock = struct('perm', rock.perm(ix), ...
                 'poro', rock.poro(ix))
end

function ix = define_subset(varargin)
   [Nx, Ny, Nz] = deal(128, 64, 1);
                                          % ()
   [I, J, K]    = deal(1:Nx, 1:Ny, 1:Nz); % Default (entire dataset)

   if nargin == 1,
                                          % (layers)
      K = varargin{1};                    % Caller specified ind. layers

   elseif nargin == 3,
                                          % (I, J, K)
      [I, J, K] = deal(varargin{:});      % Caller specified box

   elseif nargin ~= 0,
      error(['Syntax is\n\t'              , ...
             'rock = %s         %% or\n\t', ...
             'rock = %s(layers) %% or\n\t', ...
             'rock = %s(I, J, K)'], mfilename, mfilename, mfilename);
   end

   validate_range(I, Nx);
   validate_range(J, Ny);
   validate_range(K, Nz);

   [I, J, K] = ndgrid(I, J, K);

   ix = sub2ind([Nx, Ny, Nz], I(:), J(:), K(:));
end

%--------------------------------------------------------------------------

function rock = load_mat_file_gt()
   data = load('vertcase3/test_new_67.mat');
   rock = data.rock; % Fa-Fa-Fa
end

%--------------------------------------------------------------------------

function validate_range(i, n)
   assert (all((0 < i) & (i <= n)), ...
           '%s outside valid range 1:%d', inputname(1), n);
end
