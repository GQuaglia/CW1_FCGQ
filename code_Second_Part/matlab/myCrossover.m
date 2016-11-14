% CW1 BIC
% Crossover which take the two best parents and mix them

function [child] = myCrossover(parent1, parent2, functionDimension)
child = rand(functionDimension, 1);
    for i = 1:ceil((functionDimension/2) -1)
        child(i) = parent1(i);
    end
    
    for j = ceil((functionDimension/2)):functionDimension
        child(j) = parent2(j);
    end
   child = parent1;
end
