@echo off
echo ========================================================
echo Pushing IMDB_BERTopic to GitHub...
echo ========================================================

gh repo create IMDB_BERTopic --public --source=. --remote=origin --push
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo If GitHub CLI is not logged in, please run:
    echo    gh auth login
    echo and then re-run this script!
)
pause
