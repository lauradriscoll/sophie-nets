function setup_figure

color = [1 1 1];


disp('adjusting graphics defaults')
disp('   figure defaults')
set(groot, 'defaultFigurePosition', [1 1 1000 1000]);
set(groot, 'defaultFigureColor', color);
disp('   line defaults');
set(groot, 'defaultLineLineWidth', 3);
set(groot, 'defaultLineMarkerSize', 10);
disp('   axes defaults');
set(groot, 'defaultaxesbox', 'off');
set(groot, 'defaultaxesfontsize', 15);
set(groot, 'defaultaxeslabelfontsize', 1.3);
set(groot, 'defaultaxestitlefontsize', 2);
set(groot, 'defaultaxesColor', color);
set(groot, 'defaultaxesFontWeight', 'bold');
set(groot, 'defaultaxesLineWidth', 3);
disp('   legend defaults');
set(groot, 'defaultlegendfontsize', 15);
set(groot, 'defaultlegendbox', 'off');
end