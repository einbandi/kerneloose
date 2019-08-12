function sig_nb=determine_sigma(Dis, k_nb, f_local)
% Choose the width of the Gaussians based on the distance to the k.th
% neighbours.
%
% Input arguments:
%   Dis         distances between the training data points
%   k_nb        number of the considered neighbour
%   f_local     flag variable: 1 for different sigma for each point, 0 for
%               mean value for all points
% Output arguments:
%   sig_nb      sigma value based on the distances

% Copyright:    This file is part of the Kernel Mapping Toolbox.
%
%               The Kernel Mapping Toolbox is distributed under the
%               GNU General Public License (version 3 or later);
%               see <http://www.gnu.org/licenses/> for details.
%
%               Copyright Andrej Gisbrecht, 2015.

N=size(Dis,1);

% determine the indices of the k.th neighbours
Dis_sorted=sort(Dis);
ind_nb=bsxfun(@plus, k_nb', sum(Dis_sorted == 0,1)); % don't count points with distance 0 as neighbors
ind_nb=bsxfun(@plus, ind_nb, (0:N-1)*N); % convert positions to indices

% return the distance to the k.th neighbours
if f_local==1
    sig_nb=Dis_sorted(ind_nb);
else
    sig_nb=mean(Dis_sorted(ind_nb),2);
end
