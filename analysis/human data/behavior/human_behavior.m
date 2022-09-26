
targetInd = stream.continuous.stimConds(:,1,2);
anti_task = stream.continuous.stimConds(:,1,1)>3;
trial_type = stream.continuous.stimConds(:,1,1);
mouse_position = stream.continuous.windowsMousePosition;
%%
for y = 1:3
figure;
for x = 1:4
subplot(2,2,x)
hold on
rule = trial_type==y | trial_type==3+y;
draw_trial = anti_task==1 & targetInd==x & rule;
plot(mouse_position(draw_trial,1),mouse_position(draw_trial,2),'.r')

draw_trial = anti_task==0 & targetInd==x & rule;
plot(mouse_position(draw_trial,1),mouse_position(draw_trial,2),'.g')
xlim([-.5 .5])
ylim([-.5 .5])
end
end

%%
figure;histogram(stream.trialDelayLength)
%%
figure;plot(stream.continuous.state)