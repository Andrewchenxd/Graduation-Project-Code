@echo off
chcp 65001 >nul
title 一键批量评估所有数据集 (RMLA/RMLB/RMLC/ADSB)

echo ================================================================
echo  一键批量评估所有数据集
echo  数据集: RMLA (11类) / RMLB (10类) / RMLC (11类) / ADSB (198类)
echo  基于 README.md 中列出的所有评估命令
echo ================================================================
echo.
echo  开始时间: %date% %time%
echo.

:: ============================================================
:: RMLA (RML2016.10a) - 11类, batchsize=64
:: ============================================================
echo ==================== RMLA 数据集 ====================

echo [RMLA-01] TSFFN ...
python evaluate.py --checkpoint ./checkpoint_TSFFN/RMLA/PWVD/pwvd_best_network_acc_best.pth --dataset RMLA --classesnum 11 --batchsize 64

echo [RMLA-02] MCLDNN ...
python evaluate.py --checkpoint ./checkpoint_MCLDNN/RMLA/PWVD/pwvd_best_network_acc_best.pth --dataset RMLA --classesnum 11 --batchsize 64

echo [RMLA-03] CNN2 ...
python evaluate.py --checkpoint ./checkpoint_CNN2/RMLA/PWVD/pwvd_best_network_acc_best.pth --dataset RMLA --classesnum 11 --batchsize 64

echo [RMLA-04] ResNet2d ...
python evaluate.py --checkpoint ./checkpoint_ResNet2d/RMLA/PWVD/pwvd_best_network_acc_best.pth --dataset RMLA --classesnum 11 --batchsize 64

echo [RMLA-05] ResNet1d ...
python evaluate.py --checkpoint ./checkpoint_ResNet1d/RMLA/PWVD/pwvd_best_network_acc_best.pth --dataset RMLA --classesnum 11 --batchsize 64

echo [RMLA-06] PETCGDNN ...
python evaluate.py --checkpoint ./checkpoint_PETCGDNN/RMLA/PWVD/pwvd_best_network_acc_best.pth --dataset RMLA --classesnum 11 --batchsize 64

echo [RMLA-07] ICAMCNET ...
python evaluate.py --checkpoint ./checkpoint_ICAMCNET/RMLA/PWVD/pwvd_best_network_acc_best.pth --dataset RMLA --classesnum 11 --batchsize 64

echo [RMLA-08] HCGDNN ...
python evaluate.py --checkpoint ./checkpoint_HCGDNN/RMLA/PWVD/pwvd_best_network_acc_best.pth --dataset RMLA --classesnum 11 --batchsize 64

echo [RMLA-09] MMNet ...
python evaluate.py --checkpoint ./checkpoint_MMNet/RMLA/PWVD/pwvd_best_network_acc_best.pth --dataset RMLA --classesnum 11 --batchsize 64

echo [RMLA-10] AWN ...
python evaluate.py --checkpoint ./checkpoint_AWN/RMLA/PWVD/pwvd_best_network_acc_best.pth --dataset RMLA --classesnum 11 --batchsize 64

echo.
echo RMLA 评估完成！
echo.

:: ============================================================
:: RMLB (RML2016.10b) - 10类, batchsize=64
:: ============================================================
echo ==================== RMLB 数据集 ====================

echo [RMLB-01] TSFFN ...
python evaluate.py --checkpoint ./checkpoint_TSFFN/RMLB/PWVD/pwvd_best_network_acc_best.pth --dataset RMLB --classesnum 10 --batchsize 64

echo [RMLB-02] MCLDNN ...
python evaluate.py --checkpoint ./checkpoint_MCLDNN/RMLB/PWVD/pwvd_best_network_acc_best.pth --dataset RMLB --classesnum 10 --batchsize 64

echo [RMLB-03] CNN2 ...
python evaluate.py --checkpoint ./checkpoint_CNN2/RMLB/PWVD/pwvd_best_network_acc_best.pth --dataset RMLB --classesnum 10 --batchsize 64

echo [RMLB-04] DAE ...
python evaluate.py --checkpoint ./checkpoint_DAE/RMLB/PWVD/pwvd_best_network_acc_best.pth --dataset RMLB --classesnum 10 --batchsize 64

echo [RMLB-05] ResNet2d ...
python evaluate.py --checkpoint ./checkpoint_ResNet2d/RMLB/PWVD/pwvd_best_network_acc_best.pth --dataset RMLB --classesnum 10 --batchsize 64

echo [RMLB-06] ResNet1d ...
python evaluate.py --checkpoint ./checkpoint_ResNet1d/RMLB/PWVD/pwvd_best_network_acc_best.pth --dataset RMLB --classesnum 10 --batchsize 64

echo [RMLB-07] PETCGDNN ...
python evaluate.py --checkpoint ./checkpoint_PETCGDNN/RMLB/PWVD/pwvd_best_network_acc_best.pth --dataset RMLB --classesnum 10 --batchsize 64

echo [RMLB-08] ICAMCNET ...
python evaluate.py --checkpoint ./checkpoint_ICAMCNET/RMLB/PWVD/pwvd_best_network_acc_best.pth --dataset RMLB --classesnum 10 --batchsize 64

echo [RMLB-09] HCGDNN ...
python evaluate.py --checkpoint ./checkpoint_HCGDNN/RMLB/PWVD/pwvd_best_network_acc_best.pth --dataset RMLB --classesnum 10 --batchsize 64

echo [RMLB-10] FLAN ...
python evaluate.py --checkpoint ./checkpoint_FLAN/RMLB/PWVD/pwvd_best_network_acc_best.pth --dataset RMLB --classesnum 10 --batchsize 64

echo [RMLB-11] LSTM ...
python evaluate.py --checkpoint ./checkpoint_LSTM/RMLB/PWVD/pwvd_best_network_acc_best.pth --dataset RMLB --classesnum 10 --batchsize 64

echo [RMLB-12] MMNet ...
python evaluate.py --checkpoint ./checkpoint_MMNet/RMLB/PWVD/pwvd_best_network_acc_best.pth --dataset RMLB --classesnum 10 --batchsize 64

echo [RMLB-13] AWN ...
python evaluate.py --checkpoint ./checkpoint_AWN/RMLB/PWVD/pwvd_best_network_acc_best.pth --dataset RMLB --classesnum 10 --batchsize 64

echo.
echo RMLB 评估完成！
echo.

:: ============================================================
:: RMLC (RML2016.10c) - 11类, batchsize=64
:: ============================================================
echo ==================== RMLC 数据集 ====================

echo [RMLC-01] TSFFN ...
python evaluate.py --checkpoint ./checkpoint_TSFFN/RMLC/PWVD/pwvd_best_network_acc_best.pth --dataset RMLC --classesnum 11 --batchsize 64

echo [RMLC-02] MCLDNN ...
python evaluate.py --checkpoint ./checkpoint_MCLDNN/RMLC/PWVD/pwvd_best_network_acc_best.pth --dataset RMLC --classesnum 11 --batchsize 64

echo [RMLC-03] CNN2 ...
python evaluate.py --checkpoint ./checkpoint_CNN2/RMLC/PWVD/pwvd_best_network_acc_best.pth --dataset RMLC --classesnum 11 --batchsize 64

echo [RMLC-04] DAE ...
python evaluate.py --checkpoint ./checkpoint_DAE/RMLC/PWVD/pwvd_best_network_acc_best.pth --dataset RMLC --classesnum 11 --batchsize 64

echo [RMLC-05] ResNet2d ...
python evaluate.py --checkpoint ./checkpoint_ResNet2d/RMLC/PWVD/pwvd_best_network_acc_best.pth --dataset RMLC --classesnum 11 --batchsize 64

echo [RMLC-06] ResNet1d ...
python evaluate.py --checkpoint ./checkpoint_ResNet1d/RMLC/PWVD/pwvd_best_network_acc_best.pth --dataset RMLC --classesnum 11 --batchsize 64

echo [RMLC-07] PETCGDNN ...
python evaluate.py --checkpoint ./checkpoint_PETCGDNN/RMLC/PWVD/pwvd_best_network_acc_best.pth --dataset RMLC --classesnum 11 --batchsize 64

echo [RMLC-08] ICAMCNET ...
python evaluate.py --checkpoint ./checkpoint_ICAMCNET/RMLC/PWVD/pwvd_best_network_acc_best.pth --dataset RMLC --classesnum 11 --batchsize 64

echo [RMLC-09] HCGDNN ...
python evaluate.py --checkpoint ./checkpoint_HCGDNN/RMLC/PWVD/pwvd_best_network_acc_best.pth --dataset RMLC --classesnum 11 --batchsize 64

echo [RMLC-10] FLAN ...
python evaluate.py --checkpoint ./checkpoint_FLAN/RMLC/PWVD/pwvd_best_network_acc_best.pth --dataset RMLC --classesnum 11 --batchsize 64

echo [RMLC-11] LSTM ...
python evaluate.py --checkpoint ./checkpoint_LSTM/RMLC/PWVD/pwvd_best_network_acc_best.pth --dataset RMLC --classesnum 11 --batchsize 64

echo [RMLC-12] MMNet ...
python evaluate.py --checkpoint ./checkpoint_MMNet/RMLC/PWVD/pwvd_best_network_acc_best.pth --dataset RMLC --classesnum 11 --batchsize 64

echo [RMLC-13] AWN ...
python evaluate.py --checkpoint ./checkpoint_AWN/RMLC/PWVD/pwvd_best_network_acc_best.pth --dataset RMLC --classesnum 11 --batchsize 64

echo.
echo RMLC 评估完成！
echo.

:: ============================================================
:: ADSB - 198类, batchsize=32, adsbis=True, resample=True, samplenum=10
:: ============================================================
echo ==================== ADSB 数据集 ====================

echo [ADSB-01] TSFFN ...
python evaluate.py --checkpoint ./checkpoint_TSFFN/ADSB/PWVD/pwvd_best_network_acc_best.pth --dataset ADSB --classesnum 198 --batchsize 32 --adsbis True --resample True --samplenum 10

echo [ADSB-02] MCLDNN ...
python evaluate.py --checkpoint ./checkpoint_MCLDNN/ADSB/PWVD/pwvd_best_network_acc_best.pth --dataset ADSB --classesnum 198 --batchsize 32 --adsbis True --resample True --samplenum 10

echo [ADSB-03] CNN2 ...
python evaluate.py --checkpoint ./checkpoint_CNN2/ADSB/PWVD/pwvd_best_network_acc_best.pth --dataset ADSB --classesnum 198 --batchsize 32 --adsbis True --resample True --samplenum 10

echo [ADSB-04] DAE ...
python evaluate.py --checkpoint ./checkpoint_DAE/ADSB/PWVD/pwvd_best_network_acc_best.pth --dataset ADSB --classesnum 198 --batchsize 32 --adsbis True --resample True --samplenum 10

echo [ADSB-05] ResNet2d ...
python evaluate.py --checkpoint ./checkpoint_ResNet2d/ADSB/PWVD/pwvd_best_network_acc_best.pth --dataset ADSB --classesnum 198 --batchsize 32 --adsbis True --resample True --samplenum 10

echo [ADSB-06] ResNet1d ...
python evaluate.py --checkpoint ./checkpoint_ResNet1d/ADSB/PWVD/pwvd_best_network_acc_best.pth --dataset ADSB --classesnum 198 --batchsize 32 --adsbis True --resample True --samplenum 10

echo [ADSB-07] PETCGDNN ...
python evaluate.py --checkpoint ./checkpoint_PETCGDNN/ADSB/PWVD/pwvd_best_network_acc_best.pth --dataset ADSB --classesnum 198 --batchsize 32 --adsbis True --resample True --samplenum 10

echo [ADSB-08] ICAMCNET ...
python evaluate.py --checkpoint ./checkpoint_ICAMCNET/ADSB/PWVD/pwvd_best_network_acc_best.pth --dataset ADSB --classesnum 198 --batchsize 32 --adsbis True --resample True --samplenum 10

echo [ADSB-09] HCGDNN ...
python evaluate.py --checkpoint ./checkpoint_HCGDNN/ADSB/PWVD/pwvd_best_network_acc_best.pth --dataset ADSB --classesnum 198 --batchsize 32 --adsbis True --resample True --samplenum 10

echo [ADSB-10] FLAN ...
python evaluate.py --checkpoint ./checkpoint_FLAN/ADSB/PWVD/pwvd_best_network_acc_best.pth --dataset ADSB --classesnum 198 --batchsize 32 --adsbis True --resample True --samplenum 10

echo [ADSB-11] LSTM ...
python evaluate.py --checkpoint ./checkpoint_LSTM/ADSB/PWVD/pwvd_best_network_acc_best.pth --dataset ADSB --classesnum 198 --batchsize 32 --adsbis True --resample True --samplenum 10

echo [ADSB-12] MMNet ...
python evaluate.py --checkpoint ./checkpoint_MMNet/ADSB/PWVD/pwvd_best_network_acc_best.pth --dataset ADSB --classesnum 198 --batchsize 32 --adsbis True --resample True --samplenum 10

echo [ADSB-13] AWN ...
python evaluate.py --checkpoint ./checkpoint_AWN/ADSB/PWVD/pwvd_best_network_acc_best.pth --dataset ADSB --classesnum 198 --batchsize 32 --adsbis True --resample True --samplenum 10

echo.
echo ADSB 评估完成！
echo.

:: ============================================================
:: 全部完成
:: ============================================================
echo ================================================================
echo  所有数据集评估完成！
echo  开始时间: 见上方
echo  结束时间: %date% %time%
echo ================================================================
pause
