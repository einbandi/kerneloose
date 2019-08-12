function Y_ose=kmap_test(X_dist_ose, Par)
% Map new data to a low-dimensional space based on the parameters learned
% on the training data.
%
% Input arguments:
%   X_dist_ose  distances between the test and train points
%   Par         parameters of the kernel mapping; see kmap_train
% Output arguments:
%   Y_ose       low-dimensional projections of the test data

% Copyright:    This file is part of the Kernel Mapping Toolbox.
%
%               The Kernel Mapping Toolbox is distributed under the
%               GNU General Public License (version 3 or later);
%               see <http://www.gnu.org/licenses/> for details.
%
%               Copyright Andrej Gisbrecht and Alexander Schulz, 2015.

% compute the kernel
Ker_ose=exp( - bsxfun(@rdivide, X_dist_ose, Par.sig_nb));
Ker_ose=bsxfun(@rdivide, Ker_ose, sum(Ker_ose,2)); % normalise

Y_ose=Ker_ose*Par.A;
Y_nan=sum(isnan(sum(Y_ose,2)));

if (Y_nan > 0)
    warning(['Number of NaN elements is ' int2str(Y_nan) ...
        '. Consider using larger sigma for training.']);
end