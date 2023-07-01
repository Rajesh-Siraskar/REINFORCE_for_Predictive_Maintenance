rem Run python code to conduct experiments
echo off
echo -------------------------------------------------------------------------------------
echo   V.4.3 -- Env. test - no longer uses same look-ahead. Use original wear threshold
echo -------------------------------------------------------------------------------------
echo(
set pydir=D:\Rajesh\ResearchLab\REINFORCE\REINFORCE_for_Predictive_Maintenance
rem SETLOCAL
cd %pydir%
echo REINFORCE_for_Predictive_Maintenance folder
echo Executing Python code - PdM_REINFORCE_V4.3.py
python PdM_REINFORCE_V4.3.py
rem ENDLOCAL
