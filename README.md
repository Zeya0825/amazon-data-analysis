# Amazon公开电商数据分析工程

作者：胡泽雅  
项目周期：2026年10月8日至11月4日  
更新日期：2026年11月3日

## 项目简介

本项目基于Amazon公开商品信息、历史评分和评论文本，完成数据清洗、探索性分析、可视化、情感与主题分析、推荐模型评估及商业建议整理。

四周工作包括：

- 第一周：检查数据质量，清洗字段并构建分析表。
- 第二周：分析类别、价格、评分及评论特征，制作Power BI交互式看板。
- 第三周：开展TextBlob情感分析、LDA主题建模和SVD推荐实验，比较阈值、随机种子及候选集设置的影响。
- 第四周：阅读低评分商品的评论，整理业务建议；汇总四周成果，编写最终报告、项目总结PPT和第四周周报，整理代码与运行说明至GitHub，并录制项目演示视频。

## 目录

- `run.py`：统一入口，支持单任务、任务组和环境检查。
- `amazon_project/common.py`：统一数据根目录、严格情感分类边界。
- `amazon_project/tasks/`：38个正式任务模块，按功能命名；保留原编号用于对应历史报告。
- `config/`：任务注册表、15个商品的阅读分析记录、4项业务行动规则。
- `reports/week1`至`week4`：各周分析产生的统计表、配置和图表。
- `docs/code_changes.csv`：43个原脚本的保留、重构或移除去向。
- `docs/validation.md`：验证记录与验证范围。
- `tests/`：导入行为、情感边界、样例去重口径的回归检查。
- `requirements.txt`：核心依赖名；优先使用原有虚拟环境。

没有打包原始数据、处理后数据、模型、SQLite数据库、虚拟环境或评论全文。
config中的阅读分析是针对固定15个商品的研究记录，不是自动发现问题的模型，不能直接套用于新增商品。

## Windows首次检查

将ZIP解压到D盘，确认文件为 `D:\Amazon_Project_Clean\run.py`。
保留原工程 `D:\Amazon_Project`，使用其现有Python环境：

```powershell
Set-Location D:\Amazon_Project_Clean
& D:\Amazon_Project\.venv\Scripts\python.exe .\run.py check --root D:\Amazon_Project
& D:\Amazon_Project\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

check不会训练模型或改写分析文件。它仅检查包版本和基础路径。
所有任务采用同一个解释器，避免误用系统Python。

## 运行单步

```powershell
& D:\Amazon_Project\.venv\Scripts\python.exe .\run.py list
& D:\Amazon_Project\.venv\Scripts\python.exe .\run.py task 43 --root D:\Amazon_Project
& D:\Amazon_Project\.venv\Scripts\python.exe .\run.py task 36 --root D:\Amazon_Project --user DEMO_NEW_USER --top 10
```

`--root`同时指定输入数据和运行输出的根目录。以上命令会把结果写入原工程相应reports路径，同名输出会重新生成；原始CSV不改动。本整理包内reports是交付参考快照，不会因指定其他root而自动同步。
如不指定root，输入和输出都使用本整理工程根目录。数据和模型缺失时会报错，不会自动联网下载。
任务在独立进程中运行，中途失败即停止；模块导入不会触发读数据、训练和写结果。

## 从原始数据复现

原始数据需自行放置：
- `data_raw/sales/amazon.csv`：有表头的商品表。
- `data_raw/reviews/ratings_Electronics (1).csv`：无表头历史评分，四列依次为user_id、product_id、rating、timestamp。

完整复现按以下顺序运行；历史评分数据量较大，可根据需要选择任务组：

```powershell
& D:\Amazon_Project\.venv\Scripts\python.exe .\run.py group prepare --root D:\Amazon_Project
& D:\Amazon_Project\.venv\Scripts\python.exe .\run.py group explore --root D:\Amazon_Project
& D:\Amazon_Project\.venv\Scripts\python.exe .\run.py group topics --root D:\Amazon_Project
& D:\Amazon_Project\.venv\Scripts\python.exe .\run.py group recommend --root D:\Amazon_Project
& D:\Amazon_Project\.venv\Scripts\python.exe .\run.py group business --root D:\Amazon_Project
```

prepare：04–09；explore：10–22、24、25；topics：27–30、37；recommend：31–35、38、39；business：40–43。36为按需演示。
先执行prepare和explore，再执行topics或recommend；business依赖商品表、原始商品文本及固定阅读配置。
原28单种子模型仍保留，因为29和30用其模型或主题分配；37多种子实验是稳定性分析，不直接替代它。
部分任务会重建本地中间数据库和模型，因此应只在需要复现时运行对应任务组。

## 软件环境

本项目本地运行环境：Python 3.13.5、pandas 3.0.6、numpy 2.5.4、matplotlib 3.11.2、wordcloud 1.9.6、textblob 0.20.1、gensim 4.4.0、scipy 1.18.1、scikit-surprise 1.1.5。

`requirements.txt`列出核心依赖。更换环境后，建议重新检查文本处理和模型结果。

整理版工程已通过环境与路径检查、4项回归测试，以及第四周业务任务40—43的运行核对。完整历史流程尚未在整理版工程中重新运行。
Power BI文件属于独立工具产物；本包不含PBIX，原文件在reports/week2/Amazon_Week2_Dashboard.pbix，报告展示可使用先前导出的PDF。

## 分析口径

商品评分不是单条评论评分，评分人数不是销量，样例文本不是独立购买记录。历史评分与商品表只匹配6个商品，不用于强行合并整体建模。
情感分类严格使用大于/小于阈值，CSV得分用round_trip读取；接近中性不能直接称为客观中立。
推荐数据按活跃用户筛选、按用户随机留出。Top-N基于抽样候选集，不等于全商品目录或线上实验。评分误差降低不保证排序指标提高。
LDA五主题用于探索性解释，多种子实验显示主题有波动；需结合关键词和原文解释。
第四周15商品14份不同正文，重复文本只计一次；每份文本允许多个问题类型。四类正文数为12、9、6、2，不是总体问题率。
行动建议尚未执行，实际改善效果有待后续验证。

## 整理内容

移除01、02、03、23、26临时检查脚本，保留正式质量检查和浮点边界测试。移除39中未使用的catalog_counts变量。
统一路径逻辑、情感分类公共规则和运行入口；42/43的研究注释和行动规则提取到JSON配置中。保留必要中文注释、核对断言、随机种子和实验参数，保持原有分析口径。
保留独立正式任务，便于分别运行数据清洗、文本建模与推荐评估。
