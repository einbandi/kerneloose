% Kernel Mapping Toolbox.
%
% Scripts demonstrating the toolbox:
%   demo_kmap       short demo demonstrating the kernel mapping
%   demo_fisher     demo demonstrating the kernel mapping together with the
%                   estimation of the Fisher information
%
% Functions for the kernel mapping:
%   kmap_train      learns the parameters of the kernel mapping
%   kmap_test       maps new data based on the learned parameters
%   determine_sigma     estimate the width of the Gaussians based on the
%                       k.th neighbours
%
% Functions for the estimation of the Fisher information:
%   compute_fisher_distance     computes Fisher distances between sets of
%                               data points based on labelled data
%   estimate_parzen     estimates Fisher information using the Parzen
%                       window estimator
%
% Auxiliary functions:
%   README          this file
%   generate_indices    separates the data into training and test subsets

% Copyright:    This file is part of the Kernel Mapping Toolbox.
%
%               The Kernel Mapping Toolbox is distributed under the
%               GNU General Public License (version 3 or later);
%               see <http://www.gnu.org/licenses/> for details.
%
%               Copyright Andrej Gisbrecht, 2015.
