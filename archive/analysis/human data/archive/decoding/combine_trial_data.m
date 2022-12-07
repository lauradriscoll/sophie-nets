function [trial] = combine_trial_data(state, stimConds, cursorSpe)

%% pull out relevant data
contextLogic = state==20;
contextDelayLogic = state==22;
itiLogic = state==18;
goLogic = state==21;
delayLogic = state==19;
successLogic = state==5;
numTargets = 4;
stimLogic = state==17;

trial.start = find(diff([ 1; contextLogic])==1);
trial.end = find(diff([zeros(trial.start(1),1); itiLogic(trial.start(1):end)])==1);
numTrials = size(trial.start,1);
trial.stim = nan(numTrials,1);
trial.task = nan(numTrials,1);
trial.target = nan(numTrials,1);
trial.success = nan(numTrials,1);
trial.delayOn = nan(numTrials,1);
trial.go = nan(numTrials,1);
trial.moveOn = nan(numTrials,1);
trial.contextOff = nan(numTrials,1);
trial.contextOn = nan(numTrials,1);
trial.stimOn = nan(numTrials,1);

%%
for nt = 1:numTrials
    trial.stim(nt) = mode(stimConds(trial.start(nt):trial.end(nt),1,2));
    trial.task(nt) = mode(stimConds(trial.start(nt):trial.end(nt),1,1));
    trial.success(nt) = sum(successLogic(trial.start(nt):trial.end(nt)))>0;

    [moveOn, ~] = findThresholdCrossingsLowThenHigh(cursorSpe(trial.start(nt):trial.end(nt)), ...
        trial.start(nt):trial.end(nt), 2*10^(-6), 3*10^(-5));
    trial.moveOn(nt) = moveOn;
    
    if sum(contextDelayLogic(trial.start(nt):trial.end(nt)))>0
    trial.contextOn(nt) = trial.start(nt) + find(diff(contextDelayLogic(trial.start(nt):trial.end(nt)))==1) - 1;
    trial.contextOff(nt) = trial.start(nt) + find(diff(contextDelayLogic(trial.start(nt):trial.end(nt)))==-1) - 1;
    end
    
    goTime = find(goLogic(trial.start(nt):trial.end(nt))==1,1,'first');
    if ~isempty(goTime)
        
        trial.stimOn(nt) = trial.start(nt) + find(stimLogic(trial.start(nt):trial.end(nt))==1,1,'first') - 1;
        trial.go(nt) = trial.start(nt) + find(goLogic(trial.start(nt):trial.end(nt))==1,1,'first') - 1;
        trial.delayOn(nt) = trial.start(nt) + find(delayLogic(trial.start(nt):trial.end(nt))==1,1,'first') - 1;
    else
        trial.go(nt) = trial.end(nt);
    end
    
    if trial.task(nt)<4
        trial.target(nt) = trial.stim(nt);
    else
        trial.target(nt) = trial.stim(nt)+numTargets/2;
        if trial.target(nt)>numTargets
            trial.target(nt) = trial.target(nt)-numTargets;
        end
    end
    
end
end