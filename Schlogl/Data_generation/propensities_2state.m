function a = propensities_2state(x, p)
% Return reaction propensities given current state x
Cx  = x(1);
Ca  = x(2);
Cb  = x(3);

%   1. 2X + A --k1---------> 3X
%   2. 3X --k2--> 2X + A
%   3. B --k3---------> X
%   4. X --k4------------> B
a = [p.k1*Cx*Cx*Ca;
     p.k2*Cx*Cx*Cx;
     p.k3*Cb;
     p.k4*Cx;];   
end