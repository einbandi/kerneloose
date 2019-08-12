function Par=kmap_train(X_dist, Y, sig)
% Train the parameters of the kernel mapping based on the distances and
% projections of the training data.
%
% Input arguments:
%   X_dist      distances between the training data points
%   Y           low-dimensional projections of the training data
%   sig         scales the width of the Gaussians
% Output arguments:
%   Par         parameters of the trained kernel mapping;
%               Par.A   learned parameters;
%               Par.sig_nb  width of the Gaussians

% Copyright:    This file is part of the Kernel Mapping Toolbox.
%
%               The Kernel Mapping Toolbox is distributed under the
%               GNU General Public License (version 3 or later);
%               see <http://www.gnu.org/licenses/> for details.
%
%               Copyright Andrej Gisbrecht and Alexander Schulz, 2015.

% The width of the Gaussians is selected based on the distance to the k.th
% neighbour. It might be global for the whole data set or dependent on
% local structure.
k_nb    = 2; % number of neighbours
f_local = 1; % 0 for global and 1 for local widths

% compute the widths
sig_nb=determine_sigma(X_dist, k_nb, f_local);
sig_nb = sig_nb.^2;
% scale the widths to obtain more localised or more global mapping
Par.sig_nb  = sig * sig_nb ;

% compute the kernel
Ker=exp( - bsxfun(@rdivide, X_dist, Par.sig_nb));
Ker=bsxfun(@rdivide, Ker, sum(Ker,2)); % normalise

% Par.A = Ker \ Y;
Par.A = pinv(Ker) * Y;
