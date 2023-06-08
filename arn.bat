rem Run python code to conduct experiments
echo off
echo -------------------------------------------------
echo   V.2 -- Run python code to conduct experiments
echo -------------------------------------------------
echo(
set pydir=D:\Rajesh\ResearchLab\RL_for_PdM\REINFORCE_Tool_Replace_Policy
rem SETLOCAL
cd %pydir%
echo In REINFORCE_Tool_Replace_Policy folder
echo Executing Python code - PdM_REINFORCE_V3.py
python PdM_REINFORCE_V3.py
rem ENDLOCAL
