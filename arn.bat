rem Run python code to conduct experiments
echo off
echo ------------------------------------------ 
echo   Run python code to conduct experiments
echo ------------------------------------------ 
echo(
set pydir=D:\Rajesh\ResearchLab\RL_for_PdM\REINFORCE_Tool_Replace_Policy
rem SETLOCAL
cd %pydir%
echo In REINFORCE_Tool_Replace_Policy folder
echo Executing Python code - PdM-RF-AutoExpt.py
python PdM-RF-AutoExpt.py
rem ENDLOCAL