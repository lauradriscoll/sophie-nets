
second_attempt % name of run you're interested in
pm = rc.runs(1).loadPosteriorMeans();

%%

%figure;
for t = 1:500
subplot(4,1,1)
imagesc(squeeze(data.spikes(t,:,:)))
subplot(4,1,2)
imagesc((squeeze(pm.rates(:,:,t)) - squeeze(mean(pm.rates(:,:,t),2))),[-50 50])
subplot(4,1,3)
imagesc(squeeze(pm.factors(:,:,t)) - squeeze(mean(pm.factors(:,:,t),2)),[-1 1])
subplot(4,1,4)
imagesc(squeeze(pm.generator_states(:,:,t)) - squeeze(mean(pm.generator_states(:,:,t),2)),[-1 1])
pause
end

%%

%%compare to cursor data
blockData = load(fullfile(LFADS_data_dir,[filename num2str(blocks(1)) '.mat']));


%% align lfads event to behavior
[max_r, max_ri] = max(squeeze(mean(pm.rates)));
event_trial = max_r>32;
event_idx = max_ri(event_trial);

figure
hold on
plot(squeeze(mean(pm.rates(:,:,event_trial(1:100)))),'k')
plot(max_ri(event_trial),max_r(event_trial),'om')
plot(squeeze(mean(pm.rates(:,:,~event_trial(1:100)))),'b')

%%compare to cursor data
blockData = load(fullfile(LFADS_data_dir,[filename num2str(4) '.mat']));

plot(max_ri(event_trial),data.rt(event_trial),'.')
%%


figure;
hold on
shift_ind = 10;
for t = 1:80
if event_trial(t+shift_ind)
plot(blockData.allCursor{1}(t,max_ri(t+shift_ind)*2-20:max_ri(t+shift_ind)*2+20,1),blockData.allCursor{1}(t,max_ri(t+shift_ind)*2-20:max_ri(t+shift_ind)*2+20,2),':b')
plot(blockData.allCursor{1}(t,max_ri(t+shift_ind)*2,1),blockData.allCursor{1}(t,max_ri(t+shift_ind)*2,2),'.m','MarkerSize',20)
end
end

figure;
hold on
for t = 1:80
plot(blockData.allCursor{1}(t,:,1),blockData.allCursor{1}(t,:,2),':b')
if event_trial(t)
plot(blockData.allCursor{1}(t,max_ri(t)*2,1),blockData.allCursor{1}(t,max_ri(t)*2,2),'.m','MarkerSize',20)
end
end

    






