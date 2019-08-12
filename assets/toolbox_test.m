X = [ 1,2; 2,3; 3,4; 5,2; 1,6];
X_tr = [ 1,2; 2,3; 3,4 ];
X_ose = [1,2; 1,6];
[N, D]=size(X);

% select a subset of data points
##ratio=0.1; % ratio for training subset
##idx=generate_indices(Labels, ratio);
##
##% prepare the distances
##X_tr=X(idx{1},:);
##X_ose=X(idx{2},:);
X_dist_tr=squareform(pdist(X_tr)); % distances between training data points
X_dist_ose=pdist2(X_ose,X_tr) % distances from test to train data points
##

##%% compute mapping on training data
##% For simplicity the data is projected to two dimensions using classical
##% MDS. In principle, this could be an arbitrary technique generating
##% low-dimensional projections of data.
Y=cmdscale(X_dist_tr); % compute classical MDS
##Y=Y(:,[1 2]); % take the first two components
##
##% visualise the projected training data
##figure
##marker_size=20;
##scatter(Y(:,1), Y(:,2), marker_size, Labels(idx{1}), 'filled');
##
##%% train the kernel mapping
sig=0.1; % scale the width of the Gaussians
Par=kmap_train(X_dist_tr, Y, sig); % learn the parameters of the mapping
##
##%% map the remaining data
##% use the learned parameters to map the remaining data
Y_ose=kmap_test(X_dist_ose, Par)
##
##% visualise the projected test data
##figure
##marker_size=10;
##scatter(Y_ose(:,1), Y_ose(:,2), marker_size, Labels(idx{2}), 'filled');
