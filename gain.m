clc
clear all
load linear.mat
load eig1.mat
load eig2.mat
load freeinds.txt
load pinds.txt

format long
% Set some parameters
mode = 'min';  % 'min' or 'lqr'      
ne = 11;       % how many eigenvalues to compute
nes = 2;      % first pair of eigenvalues to shift by 0.25

shift = 0.25;

As = [Avs1, Ats2];
[n1,n2] = size(As);
A = [Aa,  As; 
     As', sparse(n2,n2)];
M = [Ma, sparse(n1,n2); 
     sparse(n2,n2+n1)];

nc = 2;   % number of control variables
ns1= size(Avs1,2); % no. of velocity lagrange multipliers
ns2= size(Ats2,2); % no. of temperature lagrange multipliers

%disp('Eigenvalues of A')
D1 = diag(D1)

% find unstable eig
iu = find(real(D1) > 0);
nu = length(iu);
fprintf(1, 'Number of unstable eigenvalues of A = %d\n', nu)

% select nes eigenvalues with largest real part
% NOTE: Make sure they appear in conjugate pairs
[d,ii]=sort(real(D1), 'descend');
disp('11 eigenvalues of A with largest real part')
D1 = D1(ii(1:ne))
% with shift1 first pair of D1 must be unstable
%assert(min(real(D1(ii(1:nes))+shift)) > 0.0)
Vt = Vt(:,ii(1:ne));

%disp('Eigenvalues of A^T')
D2 = diag(D2);

% find unstable eig
iu = find(real(D2) > 0);
nu2= length(iu);
%fprintf(1, 'Number of unstable eigenvalues of A^T= %d\n', nu2)
assert(nu == nu2)

% select nes eigenvalues with largest real part
[d,ii]=sort(real(D2), 'descend');
disp('11 eigenvalues of A^T with largest real part')
D2 = D2(ii(1:ne))
% with shift1 first pair of D2 must be unstable
%assert(min(real(D2(ii(1:nes))+shift)) > 0.0)
Zt = Zt(:,ii(1:ne));

% NOTE: check that eigenvalues are in same order
for j=1:ne
   assert(abs(D1(j) - D2(j)) < 1.0e-10)
   if ne == 6
      % check all eigenvalues are complex
      assert(abs(imag(D1(j))) > eps)
      Vt(:,j) = Vt(:,j) / max(abs(Vt(:,j)));
   else % check 7th eigenvalue is real
      assert(max(abs(imag(Vt(:,7)))) < 1.0e-13)
   end
end

% make Vt and Zt orthonormal
% p must be diagonal
%disp('Following must be a diagonal matrix. Is it ?')
p = Vt.' * M * Zt;
% Check p is diagonal and diagonal entries are non-zero
assert(min(abs(diag(p))) > 0.0);
assert(is_diag(p)==1)
p = diag(p);

% normalize
for j=1:ne
   Zt(:,j) = Zt(:,j) / p(j);
end

% freeinds, pinds are indices inside fenics
% We have to shift by one since python indexing starts at 0 but matlab 
% starts at 1
freeinds = freeinds + 1;
pinds    = pinds + 1;
% get matlab indices of velocity+temperature
[tmp,vTinds] = setdiff(freeinds, pinds, 'stable');
% get matlab indices of pressure
nf    = length(freeinds);
pinds = setdiff(1:nf, vTinds, 'stable');

% eigenvector component for velocity+temperature
Vty = Vt(vTinds,:);  % eigenvectors of (A,M)
Zty = Zt(vTinds,:);  % eigenvectors of (A',M')
Ztp = [Zt(pinds,:);  % pressure and other lagrange multipliers
       Zt(nf+1:end,:)];

E11 = M(vTinds,vTinds);
A11 = A(vTinds,vTinds);
A12 = [A(vTinds,pinds), As(vTinds,:)];

nvt = length(vTinds);
B1  = [sparse(nvt,2)];
B2  = [sparse(length(pinds),2);
       Bv, sparse(ns1,1);
       sparse(ns2,1), Bt];

% check orthonormality
%disp('Is this identity matrix ?')
p = Vty.' * E11 * Zty; 
% Check diagonal entries are 1
assert(max(abs(diag(p)-1.0)) < 1.0e-10);
assert(is_diag(p)==1)

% NOTE: We are assuming that unstable eigenvalues are in
% complex conjugate pairs.
U = (1/sqrt(2)) * [1,   1; ...
                   1i, -1i];
if ne==2
   U = U;
elseif ne==4
   U = blkdiag(U,U);
elseif ne==6
   U = blkdiag(U,U,U);
elseif ne==7
   U = blkdiag(U,U,U,1); 
elseif ne==11
    U = blkdiag(U,U,U,1,U,U);
else
   disp('ne is not correct')
   stop
end
assert(size(U,1) == ne)

Vy = Vty * U';
Zy = Zty * U.';
Zp = Ztp * U.';

disp('Vy and Zy must be real')
assert(max(max(abs(imag(Vy)))) < 1.0e-13)
assert(max(max(abs(imag(Zy)))) < 1.0e-13)

% Vy and Zy must be real, making sure imaginary part is close to zero
Vy = real(Vy);
Zy = real(Zy);
Zp = real(Zp);

Vy = [Vy(:,1:2)];  Zy = [Zy(:,1:2)]; % d1 case
Zp = [Zp(:,1:2)];


disp('Is this identity matrix ?')
p = Vy.' * E11 * Zy
% Check diagonal entries are 1
%assert(max(abs(diag(p)-1.0)) < 1.0e-10);
assert(is_diag(p)==1)

% Compute B12
np = length(pinds) + ns1 + ns2;
ny = length(vTinds);
%N  = [E11, A12; A12' sparse(np,np)];
%RHS= [sparse(ny,nc); B2];
%Z1 = N\RHS;
%B12= B1 + A11*Z1(1:ny,:);

% Project to unstable subspace
Au = Zy' * A11 * Vy;
Au = Au + shift*eye(size(Au));

%Bu = Zy' * B12;
Bu = -Zp' * B2;
if mode == 'min'
   % minimal norm control
   Ru = eye(nc);
   Qu = zeros(size(Au));
   fprintf(1,'Minimal norm feedback\n')
else
   % LQR problem
   Ru = diag([0.05, 0.01, 0.05]);
   Qu = Vy' * E11 * Vy;
   fprintf(1,'LQR feedback\n')
end

[Pu,L,G]=care(Au,Bu,Qu,Ru); %Solves Riccati equation
disp('Eigenvalues of projected system with feedback')
L
disp('Eigenvalues of Pu')
ePu = eig(Pu)
max_ePu = max(ePu);

B = sparse([B1; -B2]);
E11 = sparse(E11);
Z = sparse([Zy; Zp]);
Zy = sparse(Zy);
Pu = sparse(Pu);
Kt = Ru \ ((B' * Z) * Pu * (Zy' * E11));
Kt = full(Kt); % Feedback matrix
save('gain.mat','Kt')

