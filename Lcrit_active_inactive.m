%
%
%
%
%clc
%clear
% close all

% Parameters
D = 1e-3; % This is the refence D for dendrite and inactive spine
lambda=120; %This is lambda for dendrite and inactive spine
K=D/lambda^2; %This is the degradation rate throughout
cdis = 2.0;
I_factor=1.25;

%For matching to Neurn - not used here
GeomFh = 13.1962; %For matching to Neuron - not used herre
Ioh = 0.215e-4*1.125*GeomFh; % and this? What is 1.125?  


% --- Model ---
diamN = 0.2;
diamD = 5.0;
diamH = 1;

LH = 1.0;
l = 0.5;
LN = 2.0; %5

Lcrit_lambda_aa = [];
Lcrit_lambda_ai=[];
Lbase = [0.05:0.01:0.5];
% Lbase = [10:300];
Da_arr=D*[1:-.05:.1];
I0_arr=[];

% Auxilary variables for inactive - here D=D, lambda=lambda

s = lambda/2/D;
a = coth(LN/lambda) + (diamN/diamH)^2 * coth(LH/lambda);
B = cosh(LN/lambda) * a - 1/sinh(LN/lambda);
Q =  (D/lambda)*sinh(LN/lambda) * a;
Q = Q/B;
P = cosh(l/lambda)/sinh(LH/lambda);
P = P/B;

brk = coth(LH/lambda)+(diamH/diamN)^2*tanh(LN/lambda);%BB
alf = cosh(l/lambda)/(sinh(LH/lambda)*cosh(LN/lambda)*brk);%alpha

bet_new = -(lambda/D)*(cosh(l/lambda)/(sinh(LH/lambda)^2 *brk));
bet_new = bet_new*(cosh(l/lambda)-sinh(LH/lambda)*cosh((LH-l)/lambda)*brk);


%setting current to obtain bistabiltiy in isolated spine

FA=(((alf * ((diamN/diamD)^2)) *P)/((1+((lambda/(2*D))*((diamN/diamD)^2))*Q)))+(2*D/lambda)*bet_new;
FB=(((2*alf * ((diamN/diamD)^2)) *P)/((1+(lambda/(2*D))*((diamN/diamD)^2)*Q)));


for Da=Da_arr %runs over different values of Da - D for active spines
    lambda_a=sqrt(Da/K);


    %auxillary variables for active case
    a_a = coth(LN/lambda_a) + (diamN/diamH)^2 * coth(LH/lambda_a);
    B_a = cosh(LN/lambda_a) * a_a - 1/sinh(LN/lambda_a);
    Q_a =  (Da/lambda_a)*sinh(LN/lambda_a) * a_a;
    Q_a = Q_a/B_a;
    P_a = cosh(l/lambda_a)/sinh(LH/lambda_a);
    P_a = P_a/B_a;

    % Effective source in dendrite
    %Id = Ioh*(diamN/diamD)^2*P/(1-lambda/2/D*(diamN/diamD)^2*Q); %eg 26?

    brk_a = coth(LH/lambda_a)+(diamH/diamN)^2*tanh(LN/lambda_a);%BB
    alf_a = cosh(l/lambda_a)/(sinh(LH/lambda_a)*cosh(LN/lambda_a)*brk_a);%alpha

    bet_a = -(lambda_a/Da)*(cosh(l/lambda_a)/(sinh(LH/lambda_a)^2 *brk_a));
    bet_a = bet_a*(cosh(l/lambda_a)-sinh(LH/lambda_a)*cosh((LH-l)/lambda_a)*brk_a);


    %setting current to obtain bistabiltiy in isolated spine

    FA_a=(((alf_a * ((diamN/diamD)^2)) *P_a)/((1+((lambda/(2*D))*((diamN/diamD)^2))*Q_a)))+(2*D/lambda)*bet_a;
    FB_a=(((2*alf_a * ((diamN/diamD)^2)) *P_a)/((1+(lambda/(2*D))*((diamN/diamD)^2)*Q_a)));
    
    MMMa=lambda*log(1+I_factor*FB_a/FA_a);
    Lcrit_lambda_aa=[Lcrit_lambda_aa,MMMa];
   
    MMMi=lambda*log(1+I_factor*FB_a/FA);
    Lcrit_lambda_ai=[Lcrit_lambda_ai,MMMi];
   
   %FA_a/FA
  
    
end

figure(10);
F10=plot(Da_arr/D,Lcrit_lambda_aa,'LineWidth',2);
hold on
F10a=plot(Da_arr/D,Lcrit_lambda_ai,'-.','LineWidth',2);
title("Different Diffusion in Spines");
xlabel("D_a/D");
ylabel("Distance between spines (\mu m)");


