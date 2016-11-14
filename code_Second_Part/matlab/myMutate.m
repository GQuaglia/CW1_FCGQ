% CW1 BIC
% Mutation which is the sum of the parent and a little noise

function [child] = myMutate(parent, functionDimension)
child = rand(functionDimension, 1);
    for i = 1:functionDimension
        child(i) = parent(i) + (abs(parent(i))/rand);
    end
  child = parent;
end
