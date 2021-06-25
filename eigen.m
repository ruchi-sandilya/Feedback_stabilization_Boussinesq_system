% Matlab code for computing eigenvalues of linearized operator

clear all
load linear100.mat
load freeinds.txt
load pinds.txt
who

format long
% Set some parameters
mode = 'min'; % 'min' or 'lqr'
ne = 20;      % how many eigenvalues to compute

As = [Avs1, Ats2];
[n1,n2] = size(As);
A = [Aa,  As; 
     As', sparse(n2,n2)];
M = [Ma, sparse(n1,n2); 
     sparse(n2,n2+n1)];

% number of Lanczos vectors
opts.p = 50;

% Compute eigenvalues,vectors of (A,M)
[Vt,D1,flag] = eigs(A,M,ne,'SM',opts);
assert(flag==0)
save('eig1_100.mat','Vt','D1')
disp('Eigenvalues of A')
D1=diag(D1)

% find unstable eig
iu = find(real(D1) > 0);
nu = length(iu);
fprintf(1, 'Number of unstable eigenvalues of A = %d\n', nu)


% Compute eigenvalues,vectors of (A^T,M^T)
[Zt,D2,flag] = eigs(A',M',ne,'SM',opts);
assert(flag==0)
save('eig2_100.mat','Zt','D2')
disp('Eigenvalues of A^T')
D2=diag(D2)


% find unstable eig
iu = find(real(D2) > 0);
nu2= length(iu);
fprintf(1, 'Number of unstable eigenvalues of A^T= %d\n', nu2)
assert(nu == nu2)

