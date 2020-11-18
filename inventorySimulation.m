function output = inventorySimulation(D,threshold)
% initialization
n = size(D,1);
A = NaN(n,1);
X = NaN(n,1);
X(1) = 0;

for i = 1:n
    surplus = X(i)-D(i);
    if surplus < threshold
        A(i) = threshold - surplus;
    else
        A(i) = 0;
    end
    X(i+1) = max(surplus,0) + A(i);
    unsat(i) = max(D(i)-X(i),0);
end
output.sumInventory = sum(X);
output.sumUnsat = sum(unsat);
output.avgInventory = nanmean(X);
output.avgUnsat = nanmean(unsat);
output.testInventory = X(end);
output.testUnsat = unsat(end);
output.X = X;
output.A = A;
end