# LIBERO 后台下载与 replay

按用户要求恢复下载与 replay，确认下载已成功续传并完成两个文件后停止人工监控，不持续汇报。

- 固定数据版本：yifengzhu-hf/LIBERO-datasets @ f13aa24a3da8c43c7225569f28c562979fa0e35a，40 tasks，31.46 GiB raw。
- 下载：tmux `libero_raw_download_1030`。支持 .part 断点续传、镜像直连与 HF 代理切换重试；完成长度与 HDF5 基本结构检查后才发布正式文件。
- replay：tmux `libero_replay_cpu0_1030`、`libero_replay_cpu1_1030`。两个 disjoint worker，CPU llvmpipe 软件渲染，每进程 LP_NUM_THREADS=2，nice 15 / ionice best-effort priority 7；不使用 GPU。
- raw root：`/data/yhr/pretrain/xvla_official/libero_hdf5_raw`
- replay root：`/data/yhr/pretrain/xvla_official/libero_xvla_preprocessed`
- logs：`/data/yhr/output/online_filtering/logs/libero_data_background/{download,replay0,replay1}.log`
- manifest：`/data/yhr/pretrain/xvla_official/libero_download_tree.json`

完整旧文件保留；旧格式中断文件归档为 `.incomplete.legacy_TIMESTAMP`，不删除。新 replay 按 demo 提交进度，可续跑，并保存 source demo ID、source action indices、relative action、pre-action body/site pose、初始 controller goal、软件版本和图像时间/方向约定。图像保持已有格式：post-action、agent 翻转180°、wrist 原方向。未自动触发额外训练。

启动脚本 `research/scripts/run_libero_data_background.sh`；参数为 `download`，或 `replay --worker 0 --workers 2 --output REPLAY_ROOT`（另一 worker 为1）。两个 replay worker 等待新下载文件，完成40任务后自行退出。
