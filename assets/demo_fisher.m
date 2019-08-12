% This demo gives a quick reference on how to apply the kernel mapping,
% while using Fisher information to generate discriminative dimensionality
% reduction. Thereby the very costly Fisher information is computed only on
% the training set and the kernel mapping provides fast extension to the
% remaining data.

% Copyright:    This file is part of the Kernel Mapping Toolbox.
%
%               The Kernel Mapping Toolbox is distributed under the
%               GNU General Public License (version 3 or later);
%               see <http://www.gnu.org/licenses/> for details.
%
%               Copyright Andrej Gisbrecht, 2015.

%% generate training and test sets
% use a simple data set to demonstrate the toolbox
load toy_data_4D
[N, D]=size(X);

% select a subset of data points
ratio=0.5; % ratio for training subest
idx=generate_indices(Labels, ratio);

% select an even smaller subset of points from training data to learn
% the Fisher information
ratio_fisher=1; % ratio of training points used for Fisher information
idx_fisher=generate_indices(Labels(idx{1}), ratio_fisher);
% Note: here all training points are used for learning of the Fisher
% information. In practice, with a real world data set, one selects a
% sufficient amount of points, typically smaller than the training set.

% prepare the distances
X_tr=X(idx{1},:);
X_ose=X(idx{2},:);
X_dist_tr=squareform(pdist(X_tr)); % distances between training data points
X_dist_ose=pdist2(X_ose,X_tr); % distances from test to train data points

%% compute Fisher distances
% compute Fisher distances between training points; use a subset of
% training points to estimate the Fisher information
[X_fisher_dist_tr, Fisher_tr]=compute_fisher_distance(...
    X_tr(idx_fisher{1},:), Labels(idx{1}(idx_fisher{1})), X_tr, X_tr);
X_fisher_dist_tr=.5*(X_fisher_dist_tr+X_fisher_dist_tr'); % symmetrise

%% compute mapping on training data
% For simplicity the data is projected to two dimensions using classical
% MDS. In principle this could be an arbitrary technique generating
% low-dimensional projections of data.
Y=cmdscale(X_fisher_dist_tr); % compute classical MDS
Y=Y(:,[1 2]); % take the first two components

% visualise the projected training data
figure
marker_size=20;
scatter(Y(:,1), Y(:,2), marker_size, Labels(idx{1}), 'filled');

%% train the kernel mapping
sig=0.1; % scale the width of the Gaussians
Par=kmap_train(X_dist_tr, Y, sig); % learn the parameters of the mapping

%% map the remaining data
% use the learned parameters to map the remaining data
Y_ose=kmap_test(X_dist_ose, Par);

% visualise the projected test data
figure
marker_size=20;
scatter(Y_ose(:,1), Y_ose(:,2), marker_size, Labels(idx{2}), 'filled');
