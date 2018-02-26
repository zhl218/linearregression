clear all;
close all;
clc;

%%%% data is provided in matlab
load accidents
x = hwydata(:,14); %Population of states
y = hwydata(:,4); %Accidents per state
maxX = max(x);
minX = min(x);
maxY = max(y);
minY = min(y);
y = (y-maxY)/(maxY-minY);
%scatter(x, y);
m = length(y); %length of the data
theta = zeros(2, 1); %initial weights 
iterations = 1000;
eta = 0.1; %learning rate 
x = [ones(m, 1), (hwydata(:, 14)-maxX)/(maxX-minX)]; %adding ones 
J = ComputeC(x, y, theta);
[theta, allJ] = GradientDescent(x, y, theta, eta, iterations);
figure
subplot(1, 2, 1)
plot(hwydata(:,14), x*theta*(maxY-minY)+maxY, '-'); %regression 
hold on
scatter(hwydata(:,14), hwydata(:,4));
title('Regression')
hold off;
subplot(1, 2, 2)
plot(allJ, '-') %should converge 
title('Cost function')



%%%% define cost function 
function J = ComputeC(x, y, theta)
    m = length(y);
    error = (x*theta-y);
    J = error' * error/(2*m);
end 

%%%% define gradient descent function 
function [theta, allJ] = GradientDescent (x, y, theta, eta, iterations)
    m = length(y);
    allJ = zeros(iterations, 1);
    for i = 1: iterations
        error = (x*theta-y)';
        theta(1) = theta(1)-eta*(1/m)*error * x(:, 1);
        theta(2) = theta(2)-eta*(1/m)*error * x(:, 2);
        %store J
        allJ(i) = ComputeC(x, y, theta);
    end
end 

    

    
