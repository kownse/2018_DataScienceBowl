#taskkill /im ApplicationFrameHost.exe /f
#taskkill /im ShellExperienceHost.exe /f
#taskkill /IM dwm.exe /F

#python nucleus_mrcnn.py
python nucleus_classified_mrcnn.py
rundll32.exe powrprof.dll SetSuspendState 0,1,0