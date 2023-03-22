function [u] = VMD_hb(Signal,iK)
% some sample parameters for VMD
alpha = 2000; % moderate bandwidth constraint
tau = 0;      % noise-tolerance (no strict fidelity enforcement)�������ܣ�һ��ȡ0����
K = iK;        % 3 modes
DC = 0;       % no DC part imposed
init = 1;     % initialize omegas uniformly
tol = 1e-7;   %���ź�����������̶�
DData = Signal;
%--------------- Run actual VMD code

[u, u_hat, omega] = VMD(DData, alpha, tau, K, DC, init, tol);

%--------------- Visualization

% For convenience here: Order omegas increasingly and reindex u/u_hat
[~, sortIndex] = sort(omega(end,:));
omega = omega(:,sortIndex);
u_hat = u_hat(:,sortIndex);
u = u(sortIndex,:);