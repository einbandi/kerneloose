function J=estimate_parzen(X, Labels, Dis, sig_dt)
% Estimates Fisher information matrix using Parzen non-parametric estimator.
%
% Input arguments:
%   X           data points used for Parzen window estimator
%   Labels      class labels for these data points
%   Dis         distances between points at which J is estimated and the 
%               training points X
%   sig_dt      widths of the Gaussians used by the Parzen window estimator
%
% Output arguments:
%   J           Fisher information matrices on the query points
%
% For details on Fisher information please refer to:
% J. Peltonen, A. Klami and S. Kaski. Improved learning of Riemannian
% metrics for exploratory analysis. Neural Networks, 17:10877-1100, 2004

% Copyright:    This file is part of the Kernel Mapping Toolbox.
%
%               The Kernel Mapping Toolbox is distributed under the
%               GNU General Public License (version 3 or later);
%               see <http://www.gnu.org/licenses/> for details.
%
%               Copyright Andrej Gisbrecht, 2015.

% Silverman's rule of thumb
% sig_dt=1.5*1.06*size(Dis,2)^(-1/5);
Ker=exp( - bsxfun(@rdivide,Dis,2*sig_dt^2)); % N*K

[N, K]=size(Dis);
D=size(X,2);

% Cn is number of classes
C=unique(Labels);
Cn=numel(C);

% Kronecker delta d_c(c_k,c), d_c is N*Cn
% between labels of data points c_k and class labels c
d_c=zeros(K,Cn);
for ind_c=1:Cn
    d_c(:,ind_c) = C(ind_c) == Labels;
end
d_c=logical(d_c); % K*Cn

sum_Ker=sum(Ker,2); % N*1
XI=bsxfun(@rdivide,Ker,sum_Ker); % N*K
E_xi=XI*X; % N*D

E_xi_c=zeros(N,D,Cn); % N*D*Cn
for ind_c=1:Cn
    sum_psi_Ker=sum(Ker(:,d_c(:,ind_c)),2); % N*1
    psi_Ker_c=bsxfun(@times, d_c(:,ind_c)', Ker); % N*K
    XI_c=bsxfun(@rdivide, psi_Ker_c, sum_psi_Ker); % N*K
    E_xi_c(:,:,ind_c)=XI_c*X; % N*D
end

B=bsxfun(@minus, E_xi_c, E_xi); % N*D*Cn

P_cx=bsxfun(@rdivide, ...
    permute(sum( ...
        bsxfun(@times, permute(d_c, [3 1 2]), Ker) ... % N*K*Cn
    ,2), [1 3 2]), ... % N*Cn
    sum_Ker); % N*Cn

J=zeros(D,D,N);

%
B=permute(B,[2,3,1]);
for n=1:N
    B_tmp=bsxfun(@times,P_cx(n,:),B(:,:,n)); % D*D
    J(:,:,n)=B_tmp*B(:,:,n)';
    J(:,:,n)=0.5*(J(:,:,n) + J(:,:,n)');
    
    % regularisation
    J(:,:,n)=J(:,:,n)/sum(diag(J(:,:,n)));
    J(:,:,n)=J(:,:,n)+0.1*eye(D);
    %J(:,:,n)=J(:,:,n)-det(J(:,:,n))*sum(diag(pinv(J(:,:,n))));
end

%J_n_test=0;
% slow
% for n=1:N
%     BB=zeros(D,D,Cn);
%     for ind_c=1:Cn
%         BB_tmp=P_cx(n,ind_c) * B(n,:,ind_c)' * B(n,:,ind_c); % D*D
%         BB(:,:,ind_c) = 0.5*(BB_tmp + BB_tmp');
%     end
%     J_n=sum(BB,3); % D*D
%     J(:,:,n)=J_n;
% end

%J=J/(sig_dt^4);

1;
