function sol = solvePontryagin(x, T)
    % Solve the BVP using bvp4c
    xmesh = linspace(0,T,10001);                      % Mesh para resolver Pontryagin con bvp4c
    solinit = bvpinit(xmesh, @guess);               % Initial guess de la solución
    %opts = bvpset('FJacobian',@(t,y)jacobianPontryagin(t,y)); % Aditional Jacobian of Pontryagin
    opts = bvpset();
    sol = bvp4c(@(t, y) bvpfcn(t, y), @(ya, yb) bcfcn(ya, yb, x), solinit, opts);
end