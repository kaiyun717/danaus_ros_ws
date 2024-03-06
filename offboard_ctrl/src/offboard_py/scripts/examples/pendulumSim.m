hold off
clear all
close all

use_IMU_noise = 1;
angle_noise = 0.1;
gyro_noise = 0.2;
angle_beta = .999;

g = 9.81; %m/s^2
l_pendulum = 1.87; %length of pendulum, m 1.87
m_point = .06; %pendulum point mass, kg
m_rod = 0.125; %pendulum rod mass, kg
m_pendulum = m_point + m_rod; %total mass of pendulum, kg
I_pendulum = m_point*l_pendulum^2 + 1/12*m_rod*l_pendulum^2; %total inertia of pendulum
l_pendulum_eff = (m_point*l_pendulum + m_rod*l_pendulum/2)/m_pendulum;

m_quad = 0.65; %mass of quad, kg

omga_avoid = sqrt(g/l_pendulum_eff)

%Simulation Parameters
tend = 40;                  %simulation length, seconds
loopRate = 2000;            %PID controller loop rate, Hz (real-world test is higher, but 'limited' by ESC protocol talking to the motors
dt_euler = 0.00001;         %Euler simulation time step, seconds
dt_controller = 1/loopRate; %controller time step, seconds
t = 0:dt_euler:tend;        %simulation time array, seconds

%Initial Conditions
theta(1) = 12;          %initial test stand angle, degrees
theta_dot(1) = 0;      %initial test stand angular velocity, degrees/sec
x(1) = 0.0;
x_dot(1) = 0.0;

%Create step input array
angle_des_max = 0; %desired angle for creating step input, degrees
angle_des = zeros(1,length(t));
for i = 1:length(t)
    if t(i) < 18
        angle_des(i) = 0;
    elseif t(i) >=18 && t(i) <= 23 %step input between t = 18 and 23 seconds
        angle_des(i) = angle_des_max;
    else
        angle_des(i) = 0;
    end
end
angle_des = 10.*cos(0.5*t);

%Initialize some variables and arrays
t_controller = 0;               %time measured at the end of every controller run
PIDoutput = 0;
quadAngle = zeros(1,length(t)); %commanded quad angle
quadAngle_prev = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        BEGIN SIMULATION LOOP                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:length(t)
    %Run the PID controller at the specified loopRate
    if (t(i) - t_controller) >= dt_controller
        
         %'get' the IMU angle and angular velocity from euler integration
        if use_IMU_noise %apply noise to reading (real-world)
            angle_meas = theta(i) + angle_noise*(0.5 - rand(1));
            angVel_meas = theta_dot(i) + gyro_noise*(0.5 - rand(1));
        else %use the exact integrated values (idealized)
            angle_meas = theta(i);
            angVel_meas = theta_dot(i);
        end
        %PID CONTROLLER IMPLEMENTATION:
        angle_desired = angle_des(i);% - 0.0*x(i) - 0.5*x_dot(i);
        %error = angle_des(i) - angle_meas;
        %proportional = error;
        %derivative = -angVel_meas;
        %integral = integral + error*dt_controller;
        error = angle_meas - angle_desired;%angle_des(i);
        PIDoutput = 10*error + 13.0*angVel_meas;%Kp*proportional + Kd*derivative + Ki*integral; %summed PID output
        PIDoutput = bound(PIDoutput,-45,45);
        
        t_controller = t(i); %update time at end of PID loop so loop will not run until dt_controller has elapsed
    end
    
    %STORE PID OUTPUT / QUAD COMMANDED ANGLE
    quadAngle(i) = (1 - angle_beta)*-PIDoutput + angle_beta*quadAngle_prev; %this is quad angle with phase delay
    quadAngle_prev = quadAngle(i);
    
    %INTEGRATE THE EQUATION OF MOTION FOR THE QUAD
    Fx = -m_quad*g*sind(quadAngle(i)); %x-comp of rotor force that causes sideward acceleration - MAYBE CHANGE SO VERTICAL COMP IS ALWAYS  = M_QUAD*G
    Fx_stored(i) = -m_quad*g*sind(quadAngle(i));
    Fy_stored(i) = -m_quad*g*cosd(quadAngle(i));
    
    c = 0.1;
    x_ddot(i) = (Fx - c*x_dot(i))/m_quad; %EOM solved for theta double dot
    x(i+1) = x(i)+x_dot(i)*dt_euler; %integrate theta
    x_dot(i+1) = x_dot(i)+x_ddot(i)*dt_euler; %integrate theta dot
    
    %CALCULATE THE TORQUE ON THE PENDULUM
    alpha_quad = -x_ddot(i)*l_pendulum_eff*cosd(theta(i));
    T_quad = I_pendulum*alpha_quad;
    
    %INTEGRATE THE EQUATION OF MOTION FOR THE PENDULUM USING THE EULER METHOD
    theta_ddot(i) = (m_pendulum*g*l_pendulum)/I_pendulum*sind(theta(i)) + T_quad/I_pendulum; %EOM solved for theta double dot
    theta(i+1) = theta(i)+theta_dot(i)*dt_euler; %integrate theta
    theta_dot(i+1) = theta_dot(i)+theta_ddot(i)*dt_euler; %integrate theta dot
    
end

theta(end) = []; %get rid of last index
theta_dot(end) = []; %get rid of last index
x(end) = [];
x_dot(end) = [];

%{
%Plot the step response
figure(1)
set(gca,'Fontsize',18);
hold on
plot(t, theta,'Linewidth',2)
plot(t, angle_des,'Linewidth',2)
xlim([0 tend])
xlabel('Time (s)')
ylabel('Angle (degrees)')
title('Step Response')
grid on
hold off

figure(2)
set(gca,'Fontsize',18);
plot(t, x,'Linewidth',2)
xlim([0 tend])
xlabel('Time (s)')
ylabel('x position (m)')
title('Quad')
grid on

figure(3)
Fs = 1/dt_euler;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = t(end)*1000;             % Length of signal
%t = (0:L-1)*T;        % Time vector
Y = fft(x);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;

plot(f,P1) 
title('Single-Sided Amplitude Spectrum of S(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')
xlim([0 10])
%}
%ANIMATE
figure(4)
pause(5)
for i=1:3000:length(theta)
    font = 'Arial';
    figure(4)
    clf
    hold on
    
    %the rotating rectangle of doom
    angle = quadAngle(i);
    %angle = 0;
    R = [cosd(angle) -sind(angle); sind(angle) cosd(angle)];
    recx = [-0.5 0.5 0.5 -0.5 -0.5] + x(i);
    recy = [0.1 0.1 -0.1 -0.1 0.1];
    rec = [recx; recy];
    center = repmat([x(i); 0], 1, length(recy));
    
    s = rec - center; %shift points in the plane so that the center of rotation is at the origin
    so = R*s; %apply the rotation about the origin
    vo = so + center; %shift again so the origin goes back to the desired center of rotation 
    x_rotated = vo(1,:);
    y_rotated = vo(2,:);
    
    mass_x = x(i)+l_pendulum*sind(theta(i));
    mass_y = l_pendulum*cosd(theta(i));
    dot_x = x(i);
    dot_y = 0;
    line_x = [x(i) x(i)+l_pendulum*sind(theta(i))];
    line_y = [0 l_pendulum*cosd(theta(i))];
    

    
    plot(dot_x, dot_y, 'k.','Markersize',20)
    plot(x_rotated,y_rotated,'b','Linewidth',4)
    %arrows
    X = [0.518 0.518+Fx_stored(i)/30];
    Y = [0.313   0.313];
    a1 = annotation('arrow',X,Y,'HeadLength',7*abs(Fx_stored(i))+0.1,'HeadWidth',7*abs(Fx_stored(i))+0.1,'Linewidth',1.4*abs(Fx_stored(i))+0.1,'HeadStyle','cback1');
    a1.Color ='red';
    
    X = [0.518 0.518];
    Y = [0.313   0.313-Fy_stored(i)/30];
    a2 = annotation('arrow',X,Y,'HeadLength',-7*Fy_stored(i),'HeadWidth',-7*Fy_stored(i),'Linewidth',-1.4*Fy_stored(i),'HeadStyle','cback1');
    a2.Color ='red';
    
    plot([x(i) x(i)+sind(angle_des(i))],[0 l_pendulum*cosd(angle_des(i))],'k--')
    plot(line_x, line_y, 'k-','Linewidth',4)
    plot(mass_x,mass_y,'r.','Markersize',40)
    xlim([x(i)-1.5*l_pendulum x(i)+1.5*l_pendulum])
    ylim([-l_pendulum/2 1.5*l_pendulum])
    title(t(i))
    grid on
    axis equal
    hold off

    
    %pause(.00001)
    drawnow
end
hold off


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%helper function

function y = bound(x,bl,bu)
  %return bounded value clipped between bl and bu
  y=min(max(x,bl),bu);
end






