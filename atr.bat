rem Run python code to conduct experiments
echo off
echo -------------------------------------------------
echo   PRE-TRAINED MODEL Testing V.1.3
echo -------------------------------------------------
echo(
set pydir=D:\Rajesh\ResearchLab\RL_for_PdM\REINFORCE_Tool_Replace_Policy
rem SETLOCAL
cd %pydir%
echo Executing Python code - PdM_REINFORCE_ModelTester_V3.py
python PdM_REINFORCE_ModelTester_V3.py
rem ENDLOCAL
