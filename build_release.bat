@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo  动力滑行 v1.0.0 一键打包 (方案A 精简版)
echo ============================================
echo.

rem 清理旧构建产物
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

echo [1/4] 生成 PyInstaller 配置并构建（含资源）...
set SDL2DLL=SDL2.dll
if exist %SDL2DLL% (
    echo   - 检测到 SDL2.dll，将在打包后复制到 dist
)

".venv\Scripts\python.exe" -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --name "动力滑行" ^
    --windowed ^
    --onedir ^
    --add-data "assets\textures;assets\textures" ^
    --add-data "RX-7.glb;." ^
    --add-data "haruna_dem.npy;." ^
    --add-data "akina_1v1_track.json;." ^
    main.py

if errorlevel 1 (
    echo.
    echo [错误] 打包失败！
    pause
    exit /b 1
)

echo.
echo [2/4] 复制 SDL2.dll 到产物目录...
if exist %SDL2DLL% (
    copy /y "%SDL2DLL%" "dist\动力滑行\" >nul
    echo   - SDL2.dll 已复制
)

echo.
echo [3/4] 生成启动脚本...
echo @echo off > "dist\动力滑行\启动游戏.bat"
echo chcp 65001 ^>nul >> "dist\动力滑行\启动游戏.bat"
echo cd /d "%%~dp0" >> "dist\动力滑行\启动游戏.bat"
echo start "" "动力滑行.exe" >> "dist\动力滑行\启动游戏.bat"

echo.
echo [4/4] 打包完成！
echo.
echo   产物目录: dist\动力滑行\
echo   首次运行会自动重建 .cache 场景缓存（约1-3分钟）
echo.
pause