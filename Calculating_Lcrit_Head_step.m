
%clc
%clear
% close all

% Parameters
D = 1e-3;
cdis = 2.0;
I_factor=1.25;



% --- Model ---
diamN = 0.2;
diamD = 5.0;
diamH = 1;

LH = 1.0;
l = 0.5;
LN = 2.0;

Lcrit_lambda = [];
Lcrit_lambda_simp=[];
Lbase = [0.05:0.01:0.5];
lambda_arr=[10:10:720];
I0_arr=[]

for lambda = lambda_arr
    
    s = lambda/2/D;
    a = coth(LN/lambda) + (diamN/diamH)^2 * coth(LH/lambda);
    B = cosh(LN/lambda) * a - 1/sinh(LN/lambda);
    Q =  (D/lambda)*sinh(LN/lambda) * a;
    Q = Q/B;
    P = cosh(l/lambda)/sinh(LH/lambda);
    P = P/B;
    
    % Effective source in dendrite
    
    brk = coth(LH/lambda)+(diamH/diamN)^2*tanh(LN/lambda);%BB
    alf = cosh(l/lambda)/(sinh(LH/lambda)*cosh(LN/lambda)*brk);%alpha
    
  
    bet_new = -(lambda/D)*(cosh(l/lambda)/(sinh(LH/lambda)^2 *brk));
    bet_new = bet_new*(cosh(l/lambda)-sinh(LH/lambda)*cosh((LH-l)/lambda)*brk);
    

    %setting current to obtain bistabiltiy in isolated spine

    FA=(((alf * ((diamN/diamD)^2)) *P)/((1+((lambda/(2*D))*((diamN/diamD)^2))*Q)))+(2*D/lambda)*bet_new;
    FB=(((2*alf * ((diamN/diamD)^2)) *P)/((1+(lambda/(2*D))*((diamN/diamD)^2)*Q)));
    I0_crit = (2*D/lambda)*cdis/FA;
    I0 = I0_crit*I_factor; %should be I_factor
    I0_arr=[I0_arr,I0];
    
    %finding _Lcrit
    %prefactor=(lambda/(2*D))*(alf *((diamN/diamD)^2)*P)/(1+(lambda/(2*D))*((diamN/diamD)^2)*Q);
    
    %find L such that S(L,lambda)=cdis/(prefactor*I0)
    L_ax=[lambda/100:lambda/100:lambda];
    rhs=(2*D/lambda)*cdis/(FB*I0);
%     
    
   fun = @(L) rhs - geoser(L,lambda)
   
   LLL=fsolve(fun,lambda/5); %starting guess lambda/5
   Lcrit_lambda = [Lcrit_lambda,LLL];
   MMM=lambda*log(1+I_factor*FB/FA)
   Lcrit_lambda_simp=[Lcrit_lambda_simp,MMM];   
  
    
end

figure(10)
%F10=loglog(lambda_arr,Lcrit_lambda,'LineWidth',2) % - for confirmation,
%it's identical so don't need to plot
%hold on
F10a=loglog(lambda_arr,Lcrit_lambda_simp,'k-','LineWidth',2)
%F10a=plot(lambda_arr,Lcrit_lambda_simp,'k-','LineWidth',2)
%compare to switches in dendrites
Lcrit_dend=lambda_arr*log(1+2*I_factor);
hold on
F10b=loglog(lambda_arr,Lcrit_dend,'b--','LineWidth',2)
%F10b=plot(lambda_arr,Lcrit_dend,'b--','LineWidth',2)
title("Phase Diagram")
xlabel("\lambda (\mu m)")
ylabel("Distance between spines (\mu m)")

legend('Switch in spines', 'Switch in dendrites')


% For given value of lambda plot the bistability diagram as
% a function of L. Do this by calculating lower and upper values of Ch for
% all L values
% 

lam_ind=12; %choose which to display
lambda=lambda_arr(lam_ind)
I0=I0_arr(lam_ind);
Lcrit=Lcrit_lambda(lam_ind)

%setting aux constants

s = lambda/2/D;
a = coth(LN/lambda) + (diamN/diamH)^2 * coth(LH/lambda);
B = cosh(LN/lambda) * a - 1/sinh(LN/lambda);
Q =  (D/lambda)*sinh(LN/lambda) * a;
Q = Q/B;
P = cosh(l/lambda)/sinh(LH/lambda);
P = P/B;

% Effective source in dendrite

brk = coth(LH/lambda)+(diamH/diamN)^2*tanh(LN/lambda);%BB
alf = cosh(l/lambda)/(sinh(LH/lambda)*cosh(LN/lambda)*brk);%alpha


% There is a typo in the original text in beta - correct

bet_new = -(lambda/D)*(cosh(l/lambda)/(sinh(LH/lambda)^2 *brk));
bet_new = bet_new*(cosh(l/lambda)-sinh(LH/lambda)*cosh((LH-l)/lambda)*brk);



FA=(((alf * ((diamN/diamD)^2)) *P)/((1+((lambda/(2*D))*((diamN/diamD)^2))*Q)))+(2*D/lambda)*bet_new;
FB=(((2*alf * ((diamN/diamD)^2)) *P)/((1+(lambda/(2*D))*((diamN/diamD)^2)*Q)));

    
% %  
% % % 
L_arr=[Lcrit/2:Lcrit/40:Lcrit*2];

Ch_up=[]; 
Ch_down=[];
%  
for L=L_arr

    Up=(lambda/(2*D))*I0*(FA+FB*geoser(L,lambda));
    
    if L>Lcrit
        Down=(lambda/(2*D))*I0*(FB*geoser(L,lambda));
    else
        Down=Up;
    end
    
    Ch_up=[Ch_up,Up];
    Ch_down=[Ch_down,Down];  
     
end

figure(110)

plot(L_arr,Ch_up,'LineWidth',2)
hold on
plot(L_arr,Ch_down,'-.','LineWidth',2)

title(["\lambda=",num2str(lambda)])
xlabel("L (\mu m)")
ylabel("C_h")

legend('UP-state','DOWN-state')
%
% 


function X = myFzero(fun, x0, tol)
  X = []
  for x = x0
    guess = x
    val = 2*tol
    while abs(val) > tol
      [root, val, info, out] = fsolve(fun, guess);
      guess = guess + 1;
    end
    X = [X, root];
  end
end

function S= geoser(L, lam) % Infinite case start from 1
    S=exp( -L /lam) ./(1 -exp(-L/lam));
end
%save('LcritSourceInHead.mat','Lcrit_lambda')
%save('Lcrit2SourceInHead.mat','Lcrit_lambda')
