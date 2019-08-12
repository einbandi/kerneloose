function idx=generate_indices(Labels,ratio)
% Separate the data into training and test subsets. The same percentage of
% points, specified by 'ratio', is taken randomly from each class for the
% training set. The remaining points build the test set.
%
% Input arguments:
%   Labels      vector specifying class labels of data points
%   ratio       specify the percentage of points in the training set;
%               should be larger than 0 and not greater than 1
% Output arguments:
%   idx         cell array where idx{1} is the index of training set and
%               idx{2} is the index of test set

% Copyright:    This file is part of the Kernel Mapping Toolbox.
%
%               The Kernel Mapping Toolbox is distributed under the
%               GNU General Public License (version 3 or later);
%               see <http://www.gnu.org/licenses/> for details.
%
%               Copyright Andrej Gisbrecht, 2015.

if ratio <= 0 || ratio > 1
    error('The ratio of the training set should be larger than 0 and not greater than 1.');
end

idx={1,2};

idx{1,1}=[];
idx{1,2}=[];

for c=1:max(Labels)
    idx_c=find(Labels==c);
    N_c=numel(idx_c);
    idx_perm=idx_c(randperm(N_c));

    pos_cut=round(ratio*N_c);
    
    idx{1,1}=[idx{1,1}; idx_perm(1:pos_cut)];
    idx{1,2}=[idx{1,2}; idx_perm(pos_cut+1:end)];
    
end
