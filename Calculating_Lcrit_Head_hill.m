%
%
%
%
clc
clear
%close all



% Parameters
D = 1e-3;
cdis = 2.0;
I_factor=1.25;

ops = optimset ('TolX',1e-9, 'TolFun', 1e-9, 'MaxFunEvals', 1e6, 'MaxIter',1e6);
ops2 = optimset ('TolX',1e-9, 'TolFun', 1e-9, 'MaxFunEvals', 1e6, 'MaxIter',1e6);



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
lambda_arr=[20:20:720];
I0_arr=[];
I0_step_arr=[];

hill_slope=300;
%lambda =120

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
    I0_crit_step = (2*D/lambda)*cdis/FA;
    % here find I0 with hill function 
    
    
    
    
    I0_crit_range=I0_crit_step*[0.9:0.2/500:1.1];
    ln_range=length(I0_crit_range);
    ch_2=zeros(ln_range,2); %concentration in head of isolated spine
    
    
    for ii=1:ln_range
      I0=I0_crit_range(ii);
      iso_func = @(x) (-x + (lambda/(2*D))*I0*FA*hill(x,cdis,hill_slope));
      ch_2(ii,1)=fsolve(iso_func,cdis/5,optimoptions('fsolve','Display','off'));
      %ch_2(ii,1)=fsolve(iso_func,cdis/5,ops2);
      %ch_3(ii,2)=fsolve(iso_func,cdis,ops);
      UP_step=(lambda/(2*D))*(FA*I0*1.01);
      ch_2(ii,2)=fsolve(iso_func,UP_step,optimoptions('fsolve','Display','off'));
      %ch_2(ii,2)=fsolve(iso_func,UP_step,ops2);
    end
    
    i_crit_ind=find(abs(ch_2(:,2)-ch_2(:,1))>max(ch_2(:,2))/1000,1);
    I0_crit=I0_crit_range(i_crit_ind);
    step_error=100*(I0_crit-I0_crit_step)/I0_crit;
    %disp(['Error between step and hill I0_crit %',num2str(step_error)]);
    
%     figure(20)
%     plot(I0_crit_range,ch_2);
    %for now
    %I0_crit=I0_crit_step;
    
    I0 = I0_crit*I_factor; %should be I_factor
    I0_step=I0_crit_step*I_factor;
    I0_arr=[I0_arr,I0];
    I0_step_arr=[I0_step_arr,I0_step];
    
    %finding _Lcrit
    %prefactor=(lambda/(2*D))*(alf *((diamN/diamD)^2)*P)/(1+(lambda/(2*D))*((diamN/diamD)^2)*Q);
    
    %find L such that S(L,lambda)=cdis/(prefactor*I0)
    L_ax=[lambda/100:lambda/100:lambda];
    
    peak = @(x) (-x + (lambda/(2*D))*I0*FA*hill(x,cdis,hill_slope));
    cmin = fminbnd(peak,0,cdis*1.1, ops);
    rhs=(2*D/lambda)*cmin/(FB*I0) - hill(cmin,cdis,hill_slope)*FA/FB;
    fun = @(L) rhs - geoser(L,lambda);


    MMM=lambda*log(1+I_factor*FB/FA);
    Lcrit_lambda_simp=[Lcrit_lambda_simp,MMM];
    LLL=fsolve(fun,MMM, optimoptions('fsolve','Display','off'));
    Lcrit_lambda = [Lcrit_lambda,LLL];
end

figure(10)
F10=loglog(lambda_arr,Lcrit_lambda,'LineWidth',3);
hold on
F10a=loglog(lambda_arr,Lcrit_lambda_simp,'-.','LineWidth',2);
title("Phase Diagram")
xlabel("\lambda (\mu m)")
ylabel("Distance between spines (\mu m)")
% 
% 
% For given value of lambda plot the bistability diagram as
% a function of L. Do this by calculating lower and upper values of Ch for
% all L values
% 

lam_ind=6; %choose which to display
lambda=lambda_arr(lam_ind)
I0=I0_arr(lam_ind);
Lcrit=Lcrit_lambda(lam_ind);

I0_step=I0_step_arr(lam_ind);
Lcrit_step=Lcrit_lambda_simp(lam_ind);

%setting aux constants
% 
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

% 
% 
bet_new = -(lambda/D)*(cosh(l/lambda)/(sinh(LH/lambda)^2 *brk));
bet_new = bet_new*(cosh(l/lambda)-sinh(LH/lambda)*cosh((LH-l)/lambda)*brk);



FA=(((alf * ((diamN/diamD)^2)) *P)/((1+((lambda/(2*D))*((diamN/diamD)^2))*Q)))+(2*D/lambda)*bet_new;
FB=(((2*alf * ((diamN/diamD)^2)) *P)/((1+(lambda/(2*D))*((diamN/diamD)^2)*Q)));

%     
% % %  
% % % % 
L_arr=[Lcrit/2:Lcrit/400:Lcrit*2];

Ch_up=[]; 
Ch_down=[];
Ch_up_mod=[]; 
Ch_down_mod=[];
Ch_int_mod=[];
%


Lcrit_mod = 0;
for L=L_arr


    Up=(lambda/(2*D))*I0*(FA+FB*geoser(L,lambda));
    Down=(lambda/(2*D))*I0*(FB*geoser(L,lambda));
    
    % numeric solution -- include Hill function
    fun = @(x) (-x + (lambda/(2*D))*I0*(FA*hill(x,cdis,hill_slope) + FB*geoser(L,lambda)));
    Ch_up_mod = [Ch_up_mod, fsolve(fun,Up,optimoptions('fsolve','Display','off'))];
    %Ch_up_mod = [Ch_up_mod, fsolve(fun,Up, ops)];
    dn_mod = fsolve(fun, Down,optimoptions('fsolve','Display','off'));
    if (dn_mod < cdis) && (sign(fun(dn_mod+1e-6)) * sign(fun(dn_mod-1e-6)) < 0)
       Ch_down_mod = [Ch_down_mod, dn_mod];
       Ch_int_mod = [Ch_int_mod, fsolve(fun,cdis,optimoptions('fsolve','Display','off'))];
       %Ch_int_mod = [Ch_int_mod, fsolve(fun,cdis,ops)];
      if Lcrit_mod == 0
        Lcrit_mod = L - Lcrit/800;
      end
    else
      Ch_down_mod = [Ch_down_mod, NaN];
      Ch_int_mod = [Ch_int_mod, NaN];
    end
    
    Ch_up=[Ch_up,Up];
    if L < Lcrit
      Ch_down=[Ch_down,Up];
    else
      Ch_down=[Ch_down,Down];
    end
end

figure(110)

plot(L_arr,Ch_up,'-b','LineWidth',3)
hold on
plot(L_arr,Ch_up_mod,':b','LineWidth',3)
plot(L_arr,Ch_down,'-r','LineWidth',3)
plot(L_arr,Ch_down_mod,':r','LineWidth',2)
plot(L_arr,Ch_int_mod,'--k','LineWidth',2)
title("\lambda=",lambda)
xlabel("L (\mu m)")
ylabel("C_h")


Ch_up_step=[];
Ch_down_step=[];
for L=L_arr

    Up_step=(lambda/(2*D))*I0_step*(FA+FB*geoser(L,lambda));
    
    if L>Lcrit_step
        Down_step=(lambda/(2*D))*I0_step*(FB*geoser(L,lambda));
    else
        Down_step=Up_step;
    end
    
    Ch_up_step=[Ch_up_step,Up_step];
    Ch_down_step=[Ch_down_step,Down_step];  
     
end
figure(110)
plot(L_arr(1:20:length(L_arr)),Ch_up_step(1:20:length(L_arr)),'b+','LineWidth',3)
hold on
plot(L_arr(1:20:length(L_arr)),Ch_down_step(1:20:length(L_arr)),'rx','LineWidth',2)

% 
Lstep_err=100*(Lcrit-Lcrit_step)/Lcrit;
disp(['Lcrit step % error= ',num2str(Lstep_err)])

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

% % hill function with exponent 40
% function s = hill(x,cdis)
%     s =  x.^300./(x.^300 + cdis^300);
% end

function s = hill(x,cdis,hs)
    s =  x.^hs./(x.^hs + cdis^hs);
end

function S= geoser(L, lam) % Infinite case start from 1
    S=exp( -L /lam) ./(1 -exp(-L/lam));
end

% %save('LcritSourceInHead.mat','Lcrit_lambda')
% %save('Lcrit2SourceInHead.mat','Lcrit_lambda')
