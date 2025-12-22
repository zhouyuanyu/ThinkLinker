import os
import time, datetime, sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from codes.core.data.functions import setup_parser
from codes.core.model.lit_thinklinker import ThinkLinkerLitModule
from codes.core.data.dataset import DataModule

if __name__ == '__main__':
    start_ts = time.time()
    print(f"========== START: {datetime.datetime.fromtimestamp(start_ts).strftime('%Y-%m-%d %H:%M:%S')} ==========")
    sys.stdout.flush()

    args = setup_parser()
    pl.seed_everything(args.seed, workers=True)
    torch.set_num_threads(1)

    data_module = DataModule(args)
    lightning_model = ThinkLinkerLitModule(args)

    logger = pl.loggers.CSVLogger("./runs", name=args.run_name, flush_logs_every_n_steps=30)

    ckpt_callbacks = ModelCheckpoint(
        monitor='Val/mrr',
        save_top_k=1,
        save_weights_only=False,
        mode='max',
        filename=f'{args.run_name}-best'
    )

    early_stop_callback = EarlyStopping(
        monitor="Val/mrr", min_delta=0.00, patience=3, verbose=True, mode="max"
    )

    trainer = pl.Trainer(
        **args.trainer,
        deterministic=True,
        logger=logger,
        default_root_dir="./runs",
        callbacks=[ckpt_callbacks, early_stop_callback]
    )

    # 在第二阶段加载预训练权重
    if getattr(args, "ckpt_path", None):
        if os.path.isfile(args.ckpt_path):
            print(f"[INFO] Loading checkpoint weights from: {args.ckpt_path}")
            sys.stdout.flush()

            ckpt = torch.load(args.ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            lightning_model.load_state_dict(state_dict, strict=False)
        else:
            print(f"[WARN] --ckpt_path provided but file not found: {args.ckpt_path}. Training from scratch.")
            sys.stdout.flush()

    # 训练和验证
    trainer.fit(lightning_model, datamodule=data_module)

    # 测试
    trainer.test(lightning_model, datamodule=data_module, ckpt_path='best')

    end_ts = time.time()
    dur = int(end_ts - start_ts)
    h, m, s = dur // 3600, (dur % 3600) // 60, dur % 60
    print(
        f"========== END: {datetime.datetime.fromtimestamp(end_ts).strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Duration: {h:02d}:{m:02d}:{s:02d} ({dur} seconds) =========="
    )
    sys.stdout.flush()
