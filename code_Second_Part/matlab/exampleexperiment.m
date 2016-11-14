% runs an entire experiment for benchmarking MY_OPTIMIZER
% on the noise-free testbed. fgeneric.m and benchmarks.m
% must be in the path of Matlab/Octave
% CAPITALIZATION indicates code adaptations to be made

addpath('D:\Heriot Watt University\Bio-inspired computation\Coursework 1\bbob.v15.03\matlab');  % should point to fgeneric.m etc.
datapath = 'PUT_MY_BBOB_DATA_PATH';  % different folder for each experiment
% opt.inputFormat = 'row';
opt.algName = 'PUT ALGORITHM NAME';
opt.comments = 'PUT MORE DETAILED INFORMATION, PARAMETER SETTINGS ETC';
maxfunevals = '1 * dim'; % 10*dim is a short test-experiment taking a few minutes 
                          % INCREMENT maxfunevals successively to larger value(s)
minfunevals = 'dim + 2';  % PUT MINIMAL SENSIBLE NUMBER OF EVALUATIONS for a restart
maxrestarts = 1e4;        % SET to zero for an entirely deterministic algorithm

dimensions = [2]%, 3, 5, 10, 20, 40];  % small dimensions first, for CPU reasons
functions = benchmarks('FunctionIndices');  % or benchmarksnoisy(...)
instances = [1:5, 41:50];  % 15 function instances

more off;  % in octave pagination is on by default

t0 = clock;
rand('state', sum(100 * t0)); %rng(sum(100 * t0)); 

result = zeros(24,6);      % array to gather means of result for each dimension
meanResult = zeros(1,6);

for dim = dimensions
  for ifun = functions
    for iinstance = instances
      fgeneric('initialize', ifun, iinstance, datapath, opt); 

      % independent restarts until maxfunevals or ftarget is reached
      for restarts = 0:maxrestarts
        if restarts > 0  % write additional restarted info
          fgeneric('restart', 'independent restart')
        end
        myOptimizer('fgeneric', dim, fgeneric('ftarget'), ... 
                     eval(maxfunevals) - fgeneric('evaluations'));
        if fgeneric('fbest') < fgeneric('ftarget') || ...
           fgeneric('evaluations') + eval(minfunevals) > eval(maxfunevals)
          break;
        end  
      end

      disp(sprintf(['  f%d in %d-D, instance %d: FEs=%d with %d restarts,' ...
                    ' fbest-ftarget=%.4e, elapsed time [h]: %.2f'], ...
                   ifun, dim, iinstance, ...
                   fgeneric('evaluations'), ...
                   restarts, ...
                   fgeneric('fbest') - fgeneric('ftarget'), ...
                   etime(clock, t0)/60/60));
       
      sumInstances = sumInstances + abs(fgeneric('fbest') - fgeneric('ftarget'));     % sum all errors 
      
      fgeneric('finalize');
      
    end
    disp(['      date and time: ' num2str(clock, ' %.0f')]);
    
    result(ifun,dim) = sumInstances/15;     %create a matrix of results
    
  end
  disp(sprintf('---- dimension %d-D done ----', dim));
  
end

for i = 1:6
    meanSum =0;
    for j = 1:24
    meanSum = meanSum + result(j, i);
    end
    meanResult(i) = meanSum/24;
end

disp('Each value is the sum of the error (fbest-ftarget) calculated for each function and divided by 24, each column is a different dimension:')

disp(meanResult)
