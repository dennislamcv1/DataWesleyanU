libname report "E:\coursera\data analysis specialization\Data Analysis & Interpretation Capstone\Data sets";

ods graphics on;

proc means n mean std min max skewness kurtosis data=report.sample_report;
var manuf_lead num_units prod_steps sleep_hrs shift_hrs;
run;
proc freq data=report.sample_report; tables equip_fail trainee;
run;


proc format;
value yesno 1='Yes' 0='No';
run;

proc sgscatter data=report.sample_report;
title 'Figure 1. Association Between Quantitative Predictors and Manufacturing Lead Time';
plot manuf_lead*(num_units prod_steps sleep_hrs shift_hrs); 
run; 

proc sgplot data=report.sample_report;
   title1 "Association between Equipment Failure and Manufacturing Lead Time";
   vbox manuf_lead / category=equip_fail;
   format equip_fail yesno.;
run;
proc sgplot data=report.sample_report;
   title1 "Association between Trainee Involvement in Production and Manufacturing Lead Time";
   vbox manuf_lead / category=trainee;
   format equip_fail yesno.;
run;

proc surveyselect data=report.sample_report out=traintest seed = 123
 samprate=0.6 method=srs outall;
run;   

* lasso multiple regression with lars algorithm k=10 fold validation;
proc glmselect data=traintest plots=all seed=123;
     partition ROLE=selected(train='1' test='0');
     model manuf_lead = num_units equip_fail prod_steps trainee sleep_hrs shift_hrs/ 
     selection=lar(choose=cv stop=none) cvmethod=random(10);
run;


