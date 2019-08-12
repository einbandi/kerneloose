function [Dis_fisher, J_2]=compute_fisher_distance(Data, Labels, Data_1, Data_2)
% Computes the Fisher distances between two data sets, estimating the class
% probabilities on labelled data with Parzen window estimator.
%
% Input arguments:
%   Data        data points used for Parzen window estimator
%   Labels      class labels for these data points
%   Data_1 and Data_2   the distances are computed from points in Data_1 to
%               points in Data_2; if one data sets is larger then the other,
%               than it should be Data_1 to make the computations faster
%
% Output arguments:
%   Dis_fisher  distances between the data sets
%   J_2         Fisher information matrices on the points from Data_2
%
% For details on Fisher information please reffer to:
% J. Peltonen, A. Klami and S. Kaski. Improved learning of Riemannian
% metrics for exploratory analysis. Neural Networks, 17:10877-1100, 2004

% Copyright:    This file is part of the Kernel Mapping Toolbox.
%
%               The Kernel Mapping Toolbox is distributed under the
%               GNU General Public License (version 3 or later);
%               see <http://www.gnu.org/licenses/> for details.
%
%               Copyright Andrej Gisbrecht, 2015.

% T-point approximation
% The Fisher information is estimated on only T points between each pair of
% points.
T=3;

[N, D]=size(Data);
N_1=size(Data_1,1);
N_2=size(Data_2,1);

% Use the Silverman's rule of thumb to determine the widths of the
% Gaussians used by the Parzen window estimator
sig_dt=1.06*N^(-1/5);

% Estimate the Fisher information matrix on the positions of points from
% the data set Data_2
Dis_2=pdist2(Data_2, Data);
J_2=estimate_parzen(Data, Labels, Dis_2, sig_dt);

Data_T_1=zeros(T*N_1,D);

Dis_fisher=zeros(N_1,N_2);
for j=1:N_2
    for i=1:N_1
        Data_T_1((i-1)*T+(1:T),:)=bsxfun(@plus, Data_1(i,:), ...
            bsxfun(@times, (1:T)'/(T+1), Data_2(j,:)-Data_1(i,:)));
    end        
    Dis_T=pdist2(Data_T_1,Data);
    
    % J_1 is Fisher matrix for each point, D*D*(T*N_1)
    J_1=estimate_parzen(Data, Labels, Dis_T, sig_dt);
    
    for i=1:N_1
        % J is fisher matrix for T point between i and j, D*D*T
        J=J_1(:,:,(i-1)*T+(1:T));
        Data_T=Data_T_1((i-1)*T+(1:T),:);
        
%         % ------------>-------------
%         % distance at x_i between x_i and the first intermediate point
%         Dis_fisher(i,j)=(Data_1(i,:)-Data_T(1,:))*J_1(:,:,i)*(Data(i,:)-Data_T(1,:))';
%         
%         % distances between the first intermediate points
%         for t=1:T-1
%             Dis_fisher(i,j)=Dis_fisher(i,j)+ ...
%                 (Data_T(t,:)-Data_T(t+1,:))*J(:,:,t)*(Data_T(t,:)-Data_T(t+1,:))';
%         end
%         
%         % distance between the last intermediate point and x_j
%         Dis_fisher(i,j)=Dis_fisher(i,j)+ ...
%             (Data_T(T,:)-Data_2(j,:))*J(:,:,T)*(Data_T(T,:)-Data_2(j,:))';
        
        % ------------>-------------
        % distance at x_j between x_j and the first intermediate point
        Dis_fisher(i,j)=(Data_2(j,:)-Data_T(T,:))*J_2(:,:,j)*(Data_2(j,:)-Data_T(T,:))';
        
        % distances between the first intermediate points
        for t=T:-1:2
            Dis_fisher(i,j)=Dis_fisher(i,j)+ ...
                (Data_T(t,:)-Data_T(t-1,:))*J(:,:,t)*(Data_T(t,:)-Data_T(t-1,:))';
        end
        
        % distance between the last intermediate point and x_i
        Dis_fisher(i,j)=Dis_fisher(i,j)+ ...
            (Data_T(1,:)-Data_1(i,:))*J(:,:,1)*(Data_T(1,:)-Data_1(i,:))';
    end
    if mod(j,10)==0
        fprintf('Points: %u \n', j);
    end
end
