function xbest = myOptimizer(FUN, DIM, ftarget, maxfunevals)
% optimizer(FUN, DIM, ftarget, maxfunevals)
% samples new points uniformly randomly in [-5,5]^DIM
% and evaluates them on FUN until ftarget of maxfunevals
% is reached, or until 1e8 * DIM fevals are conducted. 

  maxfunevals = min(1e8 * DIM, maxfunevals); 
  popsize = min(maxfunevals, 10000);
  fbest = inf;
  
  for iter = 1:ceil(maxfunevals/popsize)
      
    xpop = 10 * rand(DIM, popsize) - 5;      %create a random matrix of values
    
    [fvalues, idx] = sort(feval(FUN, xpop)); % evaluate the fitness of all values
    
    if fbest > fvalues(1)                    % Selection: keep the best value
        fbest = fvalues(1);
        xbest = xpop(:,idx(1));                % first best value
        ndbest = xpop(:,idx(2));               % second best value
        xcrossover = myCrossover(xbest, ndbest, DIM);     % Crossover
        [fvalues, idx] = sort(feval(FUN, xcrossover));  % fitness the crossover child
      
        if fbest > fvalues(1)                       % if crossoverchild has a good fitness
            fbest = fvalues(1);
            xbest = xpop(:,idx(1));
            xmutant = myMutate(xbest, DIM);           % Mutation: myMutate the best result
            [fvalues, idx] = sort(feval(FUN, xmutant));
            
            if fbest > fvalues(1)                    % test the crossover-mutant
                 fbest = fvalues(1);
                 xbest = xmutant(:,idx(1));             
            end    
        end
        
        xmutant = myMutate(xbest, DIM);           % if crossover went wrong, myMutate the previous best result
        [fvalues, idx] = sort(feval(FUN, xmutant));
            
        if fbest > fvalues(1)                    % test the mutant
            fbest = fvalues(1);
            xbest = xmutant(:,idx(1));
        end      

    end
    
    if feval(FUN, 'fbest') < ftarget         % COCO-task achieved
      break;                                 % (works also for noisy functions)
    end
    
  end
  
end 

  
