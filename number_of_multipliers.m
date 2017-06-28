
options.maxit = 10;
options.isreal = 1;
options.issym = 0;
options.tol = 0.001;
global count
figure;
maxiter = 1000000;
for eig_num = 1:20
    flag = 0;
    disp(strcat('for maxiter = ',num2str(maxiter),' tol = ',num2str(options.tol)));
    %while flag == 0
        %options.tol = 0.0000001 / maxiter;
        options.maxit = maxiter;
        n = 20;
        
        count = 0;
        M0 = rand(2*n);
        M1 = sparse(2*n);    
        M1(n+1:2*n,1:n) = speye(n);
        M1(1:n,n+1:2*n) = speye(n);
        fun = @(x)mat(x,M0);
        try
            [v,~,flag] = eigs(fun,2*n,-M1,eig_num,'lr',options);
        catch
            flag = 1;
        end
   
        disp(count);
    %end

    plot(eig_num,count,'.');
    hold on;
end
    
    
    