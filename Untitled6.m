
options.maxit = 10;
options.isreal = 1;
options.issym = 0;
options.tol = 0.1;
global count
for n = 10:10:1000
    disp(strcat('for n = ',num2str(n)));
    count = 0;
    M0 = rand(2*n);
    M1 = sparse(2*n);    
    M1(n+1:2*n,1:n) = speye(n);
    M1(1:n,n+1:2*n) = speye(n);
    fun = @(x)mat(x,M0);
    [v,~,flag] = eigs(fun,2*n,-M1,1,'lr',options);
    disp(count);
end
    
    
    