function P = solveRiccati(T, trajectories, d)
    % Solve the Riccati equation using ode45

    P_final = zeros(d, d);  % Terminal condition P(T) = 0.
    [~, P] = ode45(@(t,P) rhsric(t,P,trajectories), [T 0], P_final); % Solve Riccati
    P = flip(P, 1);
    

end